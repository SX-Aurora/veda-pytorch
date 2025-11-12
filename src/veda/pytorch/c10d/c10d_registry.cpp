/**
 * Code taken and adjusted from: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/Ops.cpp
 * PyTorch License: https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>
#include "../api.h"

namespace c10d {
//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> sendVE(
	at::TensorList tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	int64_t dstRank,
	int64_t tag) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::send", tensor_vec, dstRank, tag);
	return process_group->getBackend(c10::DeviceType::VE)
		->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> recv_VE(
	at::TensorList tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	int64_t srcRank,
	int64_t tag) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::recv", tensor_vec, srcRank, tag);
	return process_group->getBackend(c10::DeviceType::VE)
		->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> recv_any_source_VE(
	at::TensorList tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	int64_t tag) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::send", tensor_vec, tag);
	return process_group->getBackend(c10::DeviceType::VE)
		->recvAnysource(tensor_vec, static_cast<int>(tag));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> reduce_VE(
	at::TensorList tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	const c10::intrusive_ptr<ReduceOp>& reduce_op,
	int64_t root_rank,
	int64_t root_tensor,
#if TORCH_VERSION_ >= 20800
	bool asyncOp,
#endif
	int64_t timeout) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::reduce", tensor_vec, root_rank, root_tensor,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	return process_group->getBackend(c10::DeviceType::VE)
		->reduce(
			tensor_vec,
			ReduceOptions{
				*reduce_op.get(),
				root_rank,
				root_tensor,
				std::chrono::milliseconds(timeout),
#if TORCH_VERSION_ >= 20800
				asyncOp
#endif
			});
}

//------------------------------------------------------------------------------
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
	broadcast_VE(
		at::TensorList tensors,
		const c10::intrusive_ptr<ProcessGroup>& process_group,
		int64_t root_rank,
		int64_t root_tensor,
		bool asyncOp,
		int64_t timeout) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::broadcast", tensor_vec, root_rank,
		root_tensor, asyncOp, timeout);
	auto work = process_group->getBackend(c10::DeviceType::VE)
					->broadcast(
						tensor_vec,
						BroadcastOptions{
							root_rank,
							root_tensor,
							std::chrono::milliseconds(timeout),
							asyncOp});
	return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
		std::move(tensor_vec), work);
}

//------------------------------------------------------------------------------
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
	allreduce_VE(
		at::TensorList tensors,
		const c10::intrusive_ptr<ProcessGroup>& process_group,
		const c10::intrusive_ptr<ReduceOp>& reduce_op,
		const std::optional<at::Tensor>& sparse_indices,
#if TORCH_VERSION_ >= 20800
		bool asyncOp,
#endif
		int64_t timeout) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::reduce", tensor_vec, sparse_indices,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	auto work = process_group->getBackend(c10::DeviceType::VE)
					->allreduce(
						tensor_vec,
						AllreduceOptions{
							*reduce_op.get(),
							std::chrono::milliseconds(timeout),
#if TORCH_VERSION_ >= 20800
							asyncOp
#endif
	});
	return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
		std::move(tensor_vec), work);
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> allreduce_coalesced_VE(
	at::TensorList tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	const c10::intrusive_ptr<ReduceOp>& reduce_op,
#if TORCH_VERSION_ >= 20800
	bool asyncOp,
#endif
	int64_t timeout) {
	auto tensor_vec = tensors.vec();
	veda::pytorch::dprint("c10d::allreduce_coalesced", tensor_vec,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
	opts.reduceOp = *reduce_op.get();
	opts.timeout = std::chrono::milliseconds(timeout);
#if TORCH_VERSION_ >= 20800
	opts.asyncOp = asyncOp;
#endif
	return process_group->getBackend(c10::DeviceType::VE)
		->allreduce_coalesced(tensor_vec, opts);
}

//------------------------------------------------------------------------------
std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
	allgather_VE(
		const std::vector<std::vector<at::Tensor>>& output_tensors,
		at::TensorList input_tensors,
		const c10::intrusive_ptr<ProcessGroup>& process_group,
#if TORCH_VERSION_ >= 20800
		bool asyncOp,
#endif
		int64_t timeout) {
	auto input_tensors_vec = input_tensors.vec();
	veda::pytorch::dprint("c10d::allgather", input_tensors,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	auto work = process_group->getBackend(c10::DeviceType::VE)
					->allgather(
						const_cast<std::vector<std::vector<at::Tensor>>&>(
							output_tensors),
						input_tensors_vec,
						AllgatherOptions{
							std::chrono::milliseconds(timeout)
#if TORCH_VERSION_ >= 20800
							, asyncOp
#endif
	});
	return std::
		tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
			output_tensors, work);
}

//------------------------------------------------------------------------------
std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_VE(
	at::Tensor& output_tensor,
	at::Tensor& input_tensor,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	bool asyncOp,
	int64_t timeout) {
	veda::pytorch::dprint("c10d::_allgather_base", output_tensor,
		input_tensor, asyncOp, timeout);
	auto work = process_group->getBackend(c10::DeviceType::VE)
					->_allgather_base(
						output_tensor,
						input_tensor,
						AllgatherOptions{
							std::chrono::milliseconds(timeout), asyncOp});
	return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(
		output_tensor, work);
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> allgather_coalesced_VE(
	const std::vector<std::vector<at::Tensor>>& output_lists,
	const at::TensorList& input_list,
	const c10::intrusive_ptr<ProcessGroup>& process_group
#if TORCH_VERSION_ >= 20800
	, bool asyncOp
#endif
) {
	auto input_list_vec = input_list.vec();
	veda::pytorch::dprint("c10d::allgather_coalesced", output_lists, input_list_vec
#if TORCH_VERSION_ >= 20800
		, asyncOp
#endif
	);
	auto opts = AllgatherOptions{};
#if TORCH_VERSION_ >= 20800
	opts.asyncOp = asyncOp;
#endif
	return process_group->getBackend(c10::DeviceType::VE)
		->allgather_coalesced(
			const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
			input_list_vec,
			opts);
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_VE(
	at::TensorList outputs,
	at::TensorList inputs,
	const c10::intrusive_ptr<ProcessGroup>& process_group
#if TORCH_VERSION_ >= 20800
	, bool asyncOp
#endif
) {
	auto output_vec = outputs.vec();
	auto input_vec = inputs.vec();
	veda::pytorch::dprint("c10d::allgather_into_tensor_coalesced",
		output_vec, input_vec
#if TORCH_VERSION_ >= 20800
		, asyncOp
#endif
	);
	auto opts = AllgatherOptions{};
#if TORCH_VERSION_ >= 20800
	opts.asyncOp = asyncOp;
#endif
	return process_group->getBackend(c10::DeviceType::VE)
		->allgather_into_tensor_coalesced(output_vec, input_vec, opts);
}

//------------------------------------------------------------------------------
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
	reduce_scatter_VE(
		const at::TensorList& output_tensors,
		const std::vector<std::vector<at::Tensor>>& input_tensors,
		const c10::intrusive_ptr<ProcessGroup>& process_group,
		const c10::intrusive_ptr<ReduceOp>& reduce_op,
#if TORCH_VERSION_ >= 20800
		bool asyncOp,
#endif
		int64_t timeout) {
	auto output_tensors_vec = output_tensors.vec();
	veda::pytorch::dprint("c10d::reduce_scatter", output_tensors_vec, input_tensors,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	auto work = process_group->getBackend(c10::DeviceType::VE)
					->reduce_scatter(
						output_tensors_vec,
						const_cast<std::vector<std::vector<at::Tensor>>&>(
							input_tensors),
						ReduceScatterOptions{
							*reduce_op.get(),
							std::chrono::milliseconds(timeout)
#if TORCH_VERSION_ >= 20800
							, asyncOp
#endif
	});
	return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
		output_tensors_vec, work);
}

//------------------------------------------------------------------------------
std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_VE(
	at::Tensor& output_tensor,
	at::Tensor& input_tensor,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	const c10::intrusive_ptr<ReduceOp>& reduce_op,
	bool asyncOp,
	int64_t timeout) {
	veda::pytorch::dprint("c10d::_reduce_scatter_base", output_tensor,
		input_tensor, asyncOp, timeout);
	auto work = process_group->getBackend(c10::DeviceType::VE)
					->_reduce_scatter_base(
						output_tensor,
						input_tensor,
						ReduceScatterOptions{
							*reduce_op.get(),
							std::chrono::milliseconds(timeout), asyncOp});
	return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(
		output_tensor, work);
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced_VE(
	at::TensorList outputs,
	at::TensorList inputs,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	const c10::intrusive_ptr<ReduceOp>& reduce_op,
#if TORCH_VERSION_ >= 20800
	bool asyncOp,
#endif
	int64_t timeout) {
	auto output_vec = outputs.vec();
	auto input_vec = inputs.vec();
	veda::pytorch::dprint("c10d::reduce_scatter_tensor_coalesced",
		output_vec, input_vec, 
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	return process_group->getBackend(c10::DeviceType::VE)
		->reduce_scatter_tensor_coalesced(
			output_vec,
			input_vec,
			ReduceScatterOptions{
				*reduce_op.get(),
				std::chrono::milliseconds(timeout)
#if TORCH_VERSION_ >= 20800
				, asyncOp
#endif
	});
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> gather_VE(
	const std::vector<std::vector<at::Tensor>>& output_tensors,
	const at::TensorList& input_tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	int64_t root_rank,
#if TORCH_VERSION_ >= 20800
	bool asyncOp,
#endif
	int64_t timeout) {
	auto input_tensors_vec = input_tensors.vec();
	veda::pytorch::dprint("c10d::gather", output_tensors, input_tensors,
		root_rank,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	return process_group->getBackend(c10::DeviceType::VE)
		->gather(
			const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
			input_tensors_vec,
			GatherOptions{
				root_rank, std::chrono::milliseconds(timeout)
#if TORCH_VERSION_ >= 20800
				, asyncOp
#endif
	});
}

//------------------------------------------------------------------------------
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_VE(
	const at::TensorList& output_tensors,
	const std::vector<std::vector<at::Tensor>>& input_tensors,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	int64_t root_rank,
	bool asyncOp,
	int64_t timeout) {
	auto output_tensors_vec = output_tensors.vec();
	veda::pytorch::dprint("c10d::gather", output_tensors_vec, input_tensors,
		root_rank, asyncOp, timeout);
	auto work =
		process_group->getBackend(c10::DeviceType::VE)
			->scatter(
				output_tensors_vec,
				const_cast<std::vector<std::vector<at::Tensor>>&>(
					input_tensors),
				ScatterOptions{
					root_rank, std::chrono::milliseconds(timeout), asyncOp});
	return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
		std::move(output_tensors_vec), work);
}

//------------------------------------------------------------------------------
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
	alltoall_VE(
		const at::TensorList& output_tensors,
		const at::TensorList& input_tensors,
		const c10::intrusive_ptr<ProcessGroup>& process_group,
#if TORCH_VERSION_ >= 20800
		bool asyncOp,
#endif
		int64_t timeout) {
	auto output_tensors_vec = output_tensors.vec();
	auto input_tensors_vec = input_tensors.vec();
	veda::pytorch::dprint("c10d::alltoall", output_tensors_vec, input_tensors_vec,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	auto work =
		process_group->getBackend(c10::DeviceType::VE)
			->alltoall(
				output_tensors_vec,
				input_tensors_vec,
				AllToAllOptions{std::chrono::milliseconds(timeout),
#if TORCH_VERSION_ >= 20800
					asyncOp
#endif
				});
	return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
		std::move(output_tensors_vec), work);
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> alltoall_base_VE(
	at::Tensor& output,
	at::Tensor& input,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	std::vector<int64_t> output_split_sizes,
	std::vector<int64_t> input_split_sizes,
#if TORCH_VERSION_ >= 20800
	bool asyncOp,
#endif
	int64_t timeout) {
	veda::pytorch::dprint("c10d::alltoall_base", output, input,
		output_split_sizes, input_split_sizes,
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif	
		timeout);
	return process_group->getBackend(c10::DeviceType::VE)
		->alltoall_base(
			output,
			input,
			output_split_sizes,
			input_split_sizes,
			AllToAllOptions{std::chrono::milliseconds(timeout)
#if TORCH_VERSION_ >= 20800
				, asyncOp
#endif
			});
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> barrierVE(
	at::Tensor /* unused */,
	const c10::intrusive_ptr<ProcessGroup>& process_group,
	const std::vector<int64_t>& device_ids,
#if TORCH_VERSION_ >= 20800
	bool asyncOp,
#endif
	int64_t timeout) {
	auto opts = BarrierOptions{};
	opts.device_ids = device_ids;
	opts.timeout = std::chrono::milliseconds(timeout);
#if TORCH_VERSION_ >= 20800
	opts.asyncOp = asyncOp;
#endif
	veda::pytorch::dprint("c10d::barrier",
#if TORCH_VERSION_ >= 20800
		asyncOp,
#endif
		timeout);
	return process_group->getBackend(c10::DeviceType::VE)->barrier(opts);
}

//------------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(c10d, VE, m) {
	m.impl("_allgather_base_",					_allgather_base_VE);
	m.impl("_reduce_scatter_base_",				_reduce_scatter_base_VE);
	m.impl("allgather_",						allgather_VE);
	m.impl("allgather_coalesced_",				allgather_coalesced_VE);
	m.impl("allgather_into_tensor_coalesced_",	allgather_into_tensor_coalesced_VE);
	m.impl("allreduce_",						allreduce_VE);
	m.impl("allreduce_coalesced_",				allreduce_coalesced_VE);
	m.impl("alltoall_",							alltoall_VE);
	m.impl("alltoall_base_",					alltoall_base_VE);
	m.impl("barrier",							barrierVE);
	m.impl("broadcast_",						broadcast_VE);
	m.impl("gather_",							gather_VE);
	m.impl("recv_",								recv_VE);
	m.impl("recv_any_source_",					recv_any_source_VE);
	m.impl("reduce_",							reduce_VE);
	m.impl("reduce_scatter_",					reduce_scatter_VE);
	m.impl("reduce_scatter_tensor_coalesced_",	reduce_scatter_tensor_coalesced_VE);
	m.impl("scatter_",							scatter_VE);
	m.impl("send",								sendVE);
}

//------------------------------------------------------------------------------
}
