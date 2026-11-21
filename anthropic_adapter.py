"""Anthropic protocol adapter for the configured local model."""
from __future__ import annotations
import json
import time
import uuid
import router as rr
TARGET_MODEL = __import__('settings').SETTINGS.get('anthropic_model') or __import__('settings').SETTINGS['default_model']
_STOP_MAP = {'stop': 'end_turn', 'length': 'max_tokens', 'tool_calls': 'tool_use'}

def _text_of(content) -> str:
    """Flatten Anthropic content (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'text':
            parts.append(block.get('text', ''))
    return '\n'.join((p for p in parts if p))

def _convert_messages(body: dict) -> list[dict]:
    out: list[dict] = []
    system = body.get('system')
    if system:
        out.append({'role': 'system', 'content': _text_of(system)})
    for msg in body.get('messages') or []:
        role = msg.get('role', 'user')
        content = msg.get('content')
        if isinstance(content, str):
            out.append({'role': role, 'content': content})
            continue
        text_parts, tool_calls, tool_results = ([], [], [])
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get('type')
            if btype == 'text':
                text_parts.append(block.get('text', ''))
            elif btype == 'tool_use':
                tool_calls.append({'id': block.get('id') or f'call_{uuid.uuid4().hex[:12]}', 'type': 'function', 'function': {'name': block.get('name', ''), 'arguments': json.dumps(block.get('input') or {})}})
            elif btype == 'tool_result':
                tool_results.append(block)
        if tool_calls and role == 'assistant':
            m: dict = {'role': 'assistant', 'tool_calls': tool_calls}
            if text_parts:
                m['content'] = '\n'.join(text_parts)
            out.append(m)
        elif tool_results and role == 'user':
            for tr in tool_results:
                out.append({'role': 'tool', 'tool_call_id': tr.get('tool_use_id', ''), 'content': _text_of(tr.get('content')) or '(empty tool result)'})
        else:
            out.append({'role': role, 'content': '\n'.join(text_parts) or ' '})
    return out

def to_openai(body: dict) -> dict:
    payload: dict = {'model': TARGET_MODEL, 'messages': _convert_messages(body), 'max_tokens': body.get('max_tokens') or 1024}
    if body.get('temperature') is not None:
        payload['temperature'] = body['temperature']
    if body.get('top_p') is not None:
        payload['top_p'] = body['top_p']
    if body.get('stop_sequences'):
        payload['stop'] = body['stop_sequences']
    if body.get('stream'):
        payload['stream'] = True
        payload['stream_options'] = {'include_usage': True}
    tools = []
    for t in body.get('tools') or []:
        if not isinstance(t, dict) or not t.get('name'):
            continue
        tools.append({'type': 'function', 'function': {'name': t['name'], 'description': t.get('description', ''), 'parameters': t.get('input_schema') or {'type': 'object', 'properties': {}}}})
    if tools:
        payload['tools'] = tools
    choice = body.get('tool_choice')
    if isinstance(choice, dict):
        ctype = choice.get('type')
        if ctype == 'auto':
            payload['tool_choice'] = 'auto'
        elif ctype == 'any':
            payload['tool_choice'] = 'required'
        elif ctype == 'tool' and choice.get('name'):
            payload['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    return payload

def _anthropic_usage(usage: dict) -> dict:
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    out = {'input_tokens': max(usage.get('prompt_tokens', 0) - cached, 0), 'output_tokens': usage.get('completion_tokens', 0)}
    if cached:
        out['cache_read_input_tokens'] = cached
    return out

def from_openai(body: dict, requested_model: str) -> dict:
    choice = (body.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    content = []
    reasoning = message.get('reasoning_content')
    if reasoning:
        content.append({'type': 'thinking', 'thinking': reasoning, 'signature': ''})
    if message.get('content'):
        content.append({'type': 'text', 'text': message['content']})
    for tc in message.get('tool_calls') or []:
        fn = tc.get('function') or {}
        try:
            args = json.loads(fn.get('arguments') or '{}')
        except Exception:
            args = {'raw': fn.get('arguments', '')}
        content.append({'type': 'tool_use', 'id': tc.get('id', ''), 'name': fn.get('name', ''), 'input': args})
    if not content:
        content.append({'type': 'text', 'text': ''})
    return {'id': 'msg_' + (body.get('id') or uuid.uuid4().hex[:24]).replace('chatcmpl-', ''), 'type': 'message', 'role': 'assistant', 'model': requested_model, 'content': content, 'stop_reason': _STOP_MAP.get(choice.get('finish_reason') or 'stop', 'end_turn'), 'stop_sequence': None, 'usage': _anthropic_usage(body.get('usage') or {})}

def _evt(event: str, data: dict) -> bytes:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'.encode()

def stream_anthropic(openai_lines, requested_model: str, start_usage: dict | None=None):
    """Translate a rr.chat_stream byte generator into Anthropic SSE events."""
    msg_id = 'msg_' + uuid.uuid4().hex[:24]
    input_tokens = 0
    output_tokens = 0
    yield _evt('message_start', {'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': requested_model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 1}}})
    block_index = -1
    open_kind = None
    tool_args_seen = ''

    def close_block(idx):
        return _evt('content_block_stop', {'index': idx})

    def open_block(idx, kind, extra):
        return _evt('content_block_start', {'index': idx, 'content_block': extra})
    stop_reason = 'end_turn'
    for raw in openai_lines:
        text = raw.decode('utf-8', 'replace').strip()
        if not text.startswith('data: '):
            continue
        chunk = text[6:].strip()
        if chunk == '[DONE]':
            break
        try:
            j = json.loads(chunk)
        except Exception:
            continue
        if j.get('error'):
            yield _evt('error', {'type': 'error', 'error': {'type': 'api_error', 'message': str(j['error'].get('message', 'upstream error'))[:400]}})
            break
        usage = j.get('usage') or {}
        if usage:
            input_tokens = usage.get('prompt_tokens', input_tokens)
            output_tokens = usage.get('completion_tokens', output_tokens)
        for choice in j.get('choices') or []:
            delta = choice.get('delta') or {}
            finish = choice.get('finish_reason')
            reasoning = delta.get('reasoning_content')
            content = delta.get('content')
            tool_calls = delta.get('tool_calls') or []
            if reasoning:
                if open_kind != 'thinking':
                    if open_kind is not None:
                        yield close_block(block_index)
                    block_index += 1
                    open_kind = 'thinking'
                    yield open_block(block_index, 'thinking', {'type': 'thinking', 'thinking': ''})
                yield _evt('content_block_delta', {'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': reasoning}})
            if content:
                if open_kind != 'text':
                    if open_kind is not None:
                        yield close_block(block_index)
                    block_index += 1
                    open_kind = 'text'
                    yield open_block(block_index, 'text', {'type': 'text', 'text': ''})
                yield _evt('content_block_delta', {'index': block_index, 'delta': {'type': 'text_delta', 'text': content}})
            for tc in tool_calls:
                fn = tc.get('function') or {}
                if fn.get('name'):
                    if open_kind is not None:
                        yield close_block(block_index)
                    block_index += 1
                    open_kind = 'tool_use'
                    tool_args_seen = ''
                    yield open_block(block_index, 'tool_use', {'type': 'tool_use', 'id': tc.get('id') or f'toolu_{uuid.uuid4().hex[:12]}', 'name': fn['name'], 'input': {}})
                args = fn.get('arguments')
                if args:
                    new = args[len(tool_args_seen):] if args.startswith(tool_args_seen) else args
                    tool_args_seen = args
                    yield _evt('content_block_delta', {'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': new}})
            if finish:
                stop_reason = _STOP_MAP.get(finish, 'end_turn')
    if open_kind is not None:
        yield close_block(block_index)
    yield _evt('message_delta', {'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})
    yield _evt('message_stop', {'type': 'message_stop'})

def count_tokens(body: dict) -> int:
    total = 0
    for msg in body.get('messages') or []:
        total += len(_text_of(msg.get('content')))
    total += len(_text_of(body.get('system')))
    for t in body.get('tools') or []:
        total += len(json.dumps(t))
    return max(total // 4, 1)
