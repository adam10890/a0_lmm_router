import inspect
from usr.plugins.a0_lmm_router.extensions.python.agent_init._10_init_servers import (
    LlamaCppInitExtension,
)

print("is coroutine function:", inspect.iscoroutinefunction(LlamaCppInitExtension.execute))
print("first source line:", inspect.getsource(LlamaCppInitExtension.execute).splitlines()[0])
