import sys, asyncio
sys.path.insert(0, "/a0")
from usr.plugins.a0_lmm_router.api import fleet_reconnect as fr
print("import OK")


class FakeReq:
    pass


h = fr.FleetReconnect.__new__(fr.FleetReconnect)
res = asyncio.run(h.process({"reset": True}, FakeReq()))
print("RESULT:", res)
