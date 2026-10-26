import json
import subprocess
from pathlib import Path
import pytest
from examples.planner_input import load_input
from tools.source_deposit import paginate,selected_pages,generate


def test_bundled_inputs_and_modes():
    root=Path(__file__).parent/'examples/data'
    for mode in ('static','dynamic'):
        config=load_input(root/(mode+'.json'),mode);assert config['solver']['time_limit']==.5


def test_deposit_exact_lines_no_padding_and_selection():
    files=[('a.py',b'a\nb\n'),('b.py',b'c\n')];pages=paginate(files,2)
    assert pages==[[('a.py',1,'a'),('a.py',2,'b')],[('b.py',1,'c')]]
    assert selected_pages(59)==list(range(59)) and selected_pages(60)==list(range(60))
    assert selected_pages(62)==list(range(30))+list(range(32,62))


def test_deposit_rejects_checkout_output_and_nonimmutable_sha(tmp_path):
    with pytest.raises(ValueError):generate(tmp_path,'a'*40,tmp_path/'inside')
    with pytest.raises(ValueError):generate(tmp_path,'HEAD',tmp_path.parent/'outside')


def test_deposit_nested_repo_cannot_write_into_parent_checkout(tmp_path):
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    nested=tmp_path/'nested';nested.mkdir()
    with pytest.raises(ValueError):generate(nested,'a'*40,tmp_path/'artifact')
