#include "api.h"

/**
 * taken from: https://dev-discuss.pytorch.org/t/backend-fallbacks/195
 * 
 * This is required to capture c10d.all_gather etc. within torch.compile!
 */
TORCH_LIBRARY_IMPL(_, AutogradVE, m) {
	m.fallback(torch::CppFunction::makeFallthrough());
}