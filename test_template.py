"""Offline template tests. Never uses real accounts or makes network requests."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent

class TemplateTests(unittest.TestCase):
    def test_isolated_setup_and_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for file in ROOT.glob('*.py'):
                if file.name in ('settings.py','kernel_template.py','setup_local.py','accounts.py','kaggle_client.py','server_manager.py','llama_installer.py'):
                    shutil.copyfile(file, root / file.name)
            shutil.copyfile(ROOT / 'settings.example.json', root / 'settings.example.json')
            env = {k:v for k,v in os.environ.items() if k not in ('KGCTRL_CONFIG','KAGGLE_USERNAME','KAGGLE_API_TOKEN')}
            env['HOME'] = str(root)
            # Setup runs with fake values only and no outbound network.
            code = '''from unittest.mock import patch
import runpy
answers = ['EXAMPLE_USER', 'EXAMPLE_ORG/EXAMPLE_REPOSITORY', 'EXAMPLE_FILE.gguf', 'EXAMPLE_ALIAS']
with patch('builtins.input', side_effect=answers), patch('getpass.getpass', return_value='EXAMPLE_NOT_A_REAL_TOKEN'):
    runpy.run_path('setup_local.py', run_name='__main__')
'''
            result = subprocess.run([sys.executable,'-c',code], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / '.env').exists())
            code = '''import ast, json
import server_manager as sm
spec = sm.MODELS[sm.DEFAULT_MODEL]
assert spec.hf_repo == 'EXAMPLE_ORG/EXAMPLE_REPOSITORY'
source = sm._serve_script(spec, 'EXAMPLE_TOPIC')
ast.parse(source)
assert 'EXAMPLE_ALIAS' in source and '--api-key' in source
from unittest.mock import patch
account = sm.accounts.Account(id='EXAMPLE_ACCOUNT', name='EXAMPLE_NAME', username='EXAMPLE_USER', token='EXAMPLE_NOT_A_REAL_TOKEN')
with patch.object(sm.accounts, 'load', return_value=[account]):
    work = sm.render_kernel(spec, account, 'dep-12345678')
ast.parse((work / 'serve.py').read_text())
assert 'signature' in (work / 'serve.py').read_text()
assert json.loads((work / 'kernel-metadata.json').read_text())['is_private'] == 'true'
'''
            result = subprocess.run([sys.executable,'-c',code], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = (root / '.env').read_bytes()
            result = subprocess.run([sys.executable,'setup_local.py'], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root / '.env').read_bytes(), before)

    def test_missing_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / 'settings.py', root / 'settings.py')
            env = {**os.environ, 'KGCTRL_CONFIG': str(root / 'absent.json')}
            result = subprocess.run([sys.executable,'-c','import settings'], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('setup_local.py', result.stderr)

    def test_source_compiles(self):
        for file in ROOT.glob('*.py'):
            if file.name.startswith('test_') or file.name in ('settings.py','kernel_template.py','setup_local.py'):
                ast.parse(file.read_text())

if __name__ == '__main__':
    unittest.main()
