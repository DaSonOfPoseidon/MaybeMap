# buildMap.main() reads the branding once, ahead of the build, and hands that same
# object to both the payload and the page.

import buildMap
from maybemap import branding, page, payload


def test_main_reads_the_branding_once_and_shares_it(monkeypatch):
    given = branding.Branding(title="Once")
    loads = []
    seen = {}

    def load():
        loads.append(1)
        return given

    def fake_build(*, branding=None):
        seen["build"] = branding
        return {"title": "stub"}

    def fake_write(payload_dict, *, branding=None):
        seen["write"] = branding
        seen["payload"] = payload_dict

    monkeypatch.setattr(buildMap, "load_branding", load)
    monkeypatch.setattr(payload, "build", fake_build)
    monkeypatch.setattr(page, "write", fake_write)

    buildMap.main()

    assert len(loads) == 1
    assert seen["build"] is given
    assert seen["write"] is given
    assert seen["payload"] == {"title": "stub"}
