"""Service migration refusal and rollback; no real service commands."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_gpu_installer_fixture',
    ROOT/'ops/qualified-model-fleet/install_shared_gpu.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    release=tmp_path/'release'; release.mkdir()
    comfy=tmp_path/'comfy'; comfy.mkdir()
    (comfy/'main.py').write_text('installed source')
    source=release/'comfy_sources.json'
    source.write_text(json.dumps({'main.py':hashlib.sha256((comfy/'main.py').read_bytes()).hexdigest()}))
    config=tmp_path/'fleet.json'; config.write_text('{}')
    (release/'release_manifest.json').write_text(json.dumps({
        'files':{'comfy_sources.json':hashlib.sha256(source.read_bytes()).hexdigest()},
        'existing_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}))
    unit_root=tmp_path/'units'; unit_root.mkdir()
    existing=unit_root/'comfyui.service.d/zzz-shared-gpu.conf'
    existing.parent.mkdir(); existing.write_bytes(b'[Service]\nEnvironment=EXISTING=1\n')
    states={'comfyui.service':'active','llm-inference.service':'inactive'}
    calls=[]

    def systemctl(*args):
        calls.append(args)
        if args[0]=='show':
            return '0' if 'MainPID' in args else states[args[1]]
        if args[0]=='stop': states[args[1]]='inactive'
        if args[0]=='start': states[args[1]]='active'
        return ''

    monkeypatch.setattr(installer,'SYSTEMD_ROOT',unit_root)
    monkeypatch.setattr(installer.os,'geteuid',lambda:0)
    monkeypatch.setattr(installer.pwd,'getpwnam',lambda name:SimpleNamespace(
        pw_dir=str(tmp_path/'home'),pw_uid=os.getuid(),pw_gid=os.getgid()))
    monkeypatch.setattr(installer,'systemctl',systemctl)
    monkeypatch.setattr(installer,'get',lambda url: {'queue_running':[],'queue_pending':[]}
        if url.endswith('/queue') else ({'prior':{'outputs':{}}} if url.endswith('/history') else {'system':{'comfyui_version':'fixture'}}))
    monkeypatch.setattr(installer,'wait_ready',lambda *a,**k:{})
    argv=['--release',str(release),'--comfy-root',str(comfy),'--config',str(config)]
    return SimpleNamespace(root=tmp_path,unit_root=unit_root,existing=existing,
        states=states,calls=calls,argv=argv)


def test_busy_queue_never_stops_services(machine,monkeypatch):
    monkeypatch.setattr(installer,'get',lambda url: {'queue_running':[['running']], 'queue_pending':[]}
        if url.endswith('/queue') else ({'prior':{'outputs':{}}} if url.endswith('/history') else {'system':{'comfyui_version':'fixture'}}))
    with pytest.raises(ValueError,match='queued/running'):
        installer.main(machine.argv)
    assert not any(c[0] in {'stop','start','daemon-reload'} for c in machine.calls)
    assert machine.existing.read_bytes()==b'[Service]\nEnvironment=EXISTING=1\n'


def test_failed_new_comfy_restores_overrides_and_original_services(machine,monkeypatch):
    def ready(url,*args,**kwargs):
        if ':8188/' in url: raise TimeoutError('new Comfy failed')
        return {}
    monkeypatch.setattr(installer,'wait_ready',ready)
    with pytest.raises(TimeoutError,match='new Comfy'):
        installer.main(machine.argv)
    assert machine.states=={'comfyui.service':'active','llm-inference.service':'inactive'}
    assert machine.existing.read_bytes()==b'[Service]\nEnvironment=EXISTING=1\n'
    assert not (machine.unit_root/'llm-inference.service.d/zzz-shared-gpu.conf').exists()
    receipts=list((machine.root/'home/.local/state/tare-qualified-models').glob('migration-*/result.json'))
    assert len(receipts)==1
    assert json.loads(receipts[0].read_text())['state']=='ROLLED_BACK'


def test_ready_services_use_separate_ports_and_common_lock(machine):
    installer.main(machine.argv)
    assert set(machine.states.values())=={'active'}
    text=(machine.unit_root/'llm-inference.service.d/zzz-shared-gpu.conf').read_text()
    image=machine.existing.read_text()
    assert '"8080"' in text and '"8188"' in image
    lock=str(machine.root/'home/.local/state/tare-qualified-models/gpu.lock')
    assert lock in text and lock in image
    assert '--history-snapshot' in image
    backups=list((machine.root/'home/.local/state/tare-qualified-models').glob('comfy-history-*.json'))
    assert len(backups)==1 and json.loads(backups[0].read_text())=={'prior':{'outputs':{}}}
    # The shared Windows test volume does not preserve POSIX mode bits.
    # Deployment records the real Linux state-file mode for host qualification.
    assert '--preload' not in text  # Listen before ComfyUI startup requests eviction.
