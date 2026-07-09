#include "api.h"

void backend_fallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
	op.redispatchBoxed(dispatch_keys.remove(c10::DispatchKey::AutogradVE), stack);
}

/**
 * taken from: https://dev-discuss.pytorch.org/t/backend-fallbacks/195
 * 
 * This is required to capture c10d.all_gather etc. within torch.compile!
 */
TORCH_LIBRARY_IMPL(_, AutogradVE, m) {
	m.fallback(torch::CppFunction::makeFromBoxedFunction<&backend_fallback>());
	//m.fallback(torch::CppFunction::makeFallthrough());
}