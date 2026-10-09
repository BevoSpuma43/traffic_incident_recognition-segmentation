import {execFileSync} from 'node:child_process';
import {runEditorTests} from './calibration_editor.mjs';

// Run from the repository root. No npm dependencies or browser are required.
const python = process.argv[2] ?? (process.platform === 'win32'
  ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const source = execFileSync(python, ['-c', `
import ast
from pathlib import Path
module = ast.parse(Path('src/cctv_incident/components/calibration_editor/__init__.py').read_text(encoding='utf-8'))
print(next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='JS' for t in n.targets)))
`], {encoding:'utf8'});
console.log(runEditorTests(source).map(test => 'PASS '+test).join('\n'));
