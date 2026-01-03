"""Generate an HTML report by executing the analysis notebook and exporting to HTML."""
import subprocess
from pathlib import Path

nb = Path(__file__).resolve().parents[1] / 'notebooks' / 'analysis.ipynb'
out = Path(__file__).resolve().parents[1] / 'notebooks' / 'report.html'

cmd = [
    'jupyter', 'nbconvert', '--to', 'html', '--execute', str(nb), '--output', str(out), '--ExecutePreprocessor.timeout=600'
]
print('Running:', ' '.join(cmd))
subprocess.check_call(cmd)
print('Report generated at', out)
