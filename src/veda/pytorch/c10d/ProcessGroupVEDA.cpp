/**
 * Code taken and adjusted from: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/ProcessGroupVEDA.cpp
 * Original License: https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE
 * From PyTorch:
 * 
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 * 
 * From Caffe2:
 * 
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 * 
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 * 
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 * 
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 * 
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 * 
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 * 
 * All contributions by Tri Dao:
 * Copyright (c) 2024 Tri Dao.
 * All rights reserved.
 * 
 * All contributions by Arm:
 * Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates
 * 
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 * 
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 * 
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "ProcessGroupVEDA.h"

#include <iostream>
#include <map>
#include <chrono>

#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "../api.h"

namespace c10d {

static uint64_t timer(void) {
	auto duration = std::chrono::system_clock::now().time_since_epoch();
	return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
};

#define MPI_CHECK(cmd)															\
	do {																		\
		int mpiStatus = cmd;													\
		if (mpiStatus != MPI_SUCCESS) {											\
			std::string err = "MPI error in: " + std::string(__FILE__) + ":" +	\
				std::to_string(__LINE__) +										\
				", with error code: " + std::to_string(mpiStatus);				\
			TORCH_CHECK(false, err);											\
		}																		\
	} while (0)

namespace {
//------------------------------------------------------------------------------
std::map<ReduceOp::RedOpType, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};

//------------------------------------------------------------------------------
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

//------------------------------------------------------------------------------
void checkSingleTensorHelper(const at::Tensor& tensor) {
	if (!tensor.is_contiguous()) {
		TORCH_CHECK(false, "input tensor has to be contiguous");
	}
	if (tensor.is_sparse()) {
		TORCH_CHECK(false, "input tensor has to be dense");
	}
}

//------------------------------------------------------------------------------
void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
	if (tensors.size() != 1) {
		TORCH_CHECK(
			false, "MPI process group does not support multi-GPU collectives");
	}
	checkSingleTensorHelper(tensors[0]);
}

//------------------------------------------------------------------------------
void checkSameSizeAndType(
    const at::Tensor& t_in,
    const std::vector<at::Tensor>& tensors) {
	for (const auto& tensor : tensors) {
		if ((tensor.numel() != t_in.numel()) ||
			(tensor.scalar_type() != t_in.scalar_type())) {
		TORCH_CHECK(false, "Tensors are not equal in size or data type");
		}
		checkSingleTensorHelper(tensor);
	}
}

//------------------------------------------------------------------------------
} // namespace

//------------------------------------------------------------------------------
std::vector<at::Tensor> ProcessGroupVEDA::WorkMPI::result() {
	return outputTensors_;
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupVEDA::WorkMPI::getFuture() {
	return future_;
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::WorkMPI::finishWorkMPIError(
    const std::exception_ptr& eptr) {
	future_->setError(eptr);
	finish(eptr);
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::WorkMPI::finishWorkMPI() {
	future_->markCompleted(at::IValue(outputTensors_));
	finish();
}

//------------------------------------------------------------------------------
ProcessGroupVEDA::AsyncWork::AsyncWork(
	MPI_Request request,
	std::vector<at::Tensor> outputTensors,
	const char* profilingTitle,
	const std::optional<std::vector<at::Tensor>>& inputTensors)
	: Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
		outputTensors_(std::move(outputTensors)),
		request_(request) {
	memset(&status_, 0, sizeof(status_));
}

//------------------------------------------------------------------------------
ProcessGroupVEDA::AsyncWork::~AsyncWork() {
	if (request_ != MPI_REQUEST_NULL) {
		std::cerr
			<< "Attempted destruction of AsyncWork before work has completed, "
			<< "terminating the program." << '\n';
		std::terminate();
	}
}

//------------------------------------------------------------------------------
bool ProcessGroupVEDA::AsyncWork::isCompleted() {
	if (request_ == MPI_REQUEST_NULL) {
		return true;
	}

	std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
	int flag = 0;
	MPI_CHECK(MPI_Test(&request_, &flag, &status_));
	if (request_ != MPI_REQUEST_NULL) {
		return false;
	}

	// request_ == MPI_REQUEST_NULL; the work has completed
	// Populate exception if request was not successful
	if (status_.MPI_ERROR != MPI_SUCCESS) {
		populateException();
	}

	return true;
}

//------------------------------------------------------------------------------
bool ProcessGroupVEDA::AsyncWork::isSuccess() const {
	if (request_ != MPI_REQUEST_NULL) {
		TORCH_CHECK(
			false,
			"Invalid call to AsyncWork::isSuccess before work has completed");
	}

	return status_.MPI_ERROR == MPI_SUCCESS;
}

//------------------------------------------------------------------------------
int ProcessGroupVEDA::AsyncWork::sourceRank() const {
	return status_.MPI_SOURCE;
}

//------------------------------------------------------------------------------
bool ProcessGroupVEDA::AsyncWork::wait(std::chrono::milliseconds /* unused */) {
	if (request_ == MPI_REQUEST_NULL) {
		// AsyncWork needs to manually call profiling end callbacks if they are set,
		// since it does not call ProcessGroup::finish().
		if (Work::recordFunctionEndCallback_) {
		Work::recordFunctionEndCallback_();
		Work::recordFunctionEndCallback_ = nullptr;
		}
		return true;
	}

	std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
	MPI_CHECK(MPI_Wait(&request_, &status_));
	auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

	// AsyncWork needs to manually call profiling end callbacks if they are set,
	// since it does not call ProcessGroup::finish().
	if (Work::recordFunctionEndCallback_) {
		Work::recordFunctionEndCallback_();
		Work::recordFunctionEndCallback_ = nullptr;
	}

	if (!ok) {
		populateException();
		std::rethrow_exception(exception_);
	}
#if TORCH_VERSION_ >= 20600
	if (c10d::allow_inflight_collective_as_graph_input()) {
		c10d::unregister_work(
			c10::intrusive_ptr<
				ProcessGroupVEDA::AsyncWork>::unsafe_reclaim_from_nonowning(this));
	}
#endif
	// Always return true, because abort API is not implemented.
	return true;
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::AsyncWork::abort() {
    TORCH_CHECK(false, "ProcessGroupVEDA::AsyncWork::abort not implemented.")
}

//------------------------------------------------------------------------------
std::vector<at::Tensor> ProcessGroupVEDA::AsyncWork::result() {
	return outputTensors_;
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::AsyncWork::populateException() {
	std::array<char, MPI_MAX_ERROR_STRING> buf{};
	int len = buf.size();
	MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
	exception_ =
		std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

//------------------------------------------------------------------------------
int ProcessGroupVEDA::mpiThreadSupport_ = 0;
std::mutex ProcessGroupVEDA::pgGlobalMutex_;

//------------------------------------------------------------------------------
void ProcessGroupVEDA::mpiExit() {
	std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
	MPI_CHECK(MPI_Finalize());
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::initMPIOnce() {
	// Initialize MPI environment. We only want to initialize once.
	static bool init_mpi_flag [[maybe_unused]] = []() {
		int mpi_was_initialized = 0;
		MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
		if (mpi_was_initialized == 0) {
			MPI_CHECK(MPI_Init_thread(
				nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpiThreadSupport_));
			if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
				TORCH_CHECK(
					false,
					"Used MPI implementation doesn't have the "
					"minimum level of threading support: "
					"MPI_THREAD_SERIALIZED. This is required by "
					"c10d package");
			}
			if (std::atexit(ProcessGroupVEDA::mpiExit)) {
				TORCH_CHECK(false, "Fail to register the MPI exit handler");
			}
		} else {
			TORCH_WARN_ONCE("MPI was previously initialized.");
		}
		return true;
	}();
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<ProcessGroupVEDA> ProcessGroupVEDA::createProcessGroupVEDA(
	std::vector<int> ranks)
{
	// Once initialization
	initMPIOnce();

	MPI_Comm groupComm = MPI_COMM_WORLD;
	int rank = -1;
	int size = -1;

	{
		std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

		// If no ranks are specified, assume we're creating the root group
		if (!ranks.empty()) {
			MPI_Group worldGroup{};
			MPI_Group ranksGroup{};
			MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
			MPI_CHECK(
				MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
			// `MPI_Comm_create` can be flaky in certain cases.
			// See: https://github.com/pytorch/pytorch/issues/53899
			constexpr int kMaxNumRetries = 3;
			bool groupComm_updated = false;
			MPI_Barrier(MPI_COMM_WORLD);
			for (const auto i : c10::irange(kMaxNumRetries)) {
			(void)i;
			if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
				groupComm_updated = true;
				break;
			}
			}
			MPI_CHECK(groupComm_updated);
			MPI_CHECK(MPI_Group_free(&worldGroup));
			MPI_CHECK(MPI_Group_free(&ranksGroup));
		}

		// Fetch rank and world size for this group (MPI_COMM_WORLD or new)
		if (groupComm != MPI_COMM_NULL) {
			MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
			MPI_CHECK(MPI_Comm_size(groupComm, &size));

			if (rank < 0 || size < 0) {
			TORCH_CHECK(false, "Failed to get the world_size / rank");
			}
		}
	}

	// If this process is not part of the group, we don't construct a
	// process group instance. This is in line with the semantics of the
	// other process group types.
	if (groupComm == MPI_COMM_NULL) {
		return c10::intrusive_ptr<ProcessGroupVEDA>();
	}

	return c10::make_intrusive<ProcessGroupVEDA>(rank, size, groupComm);
}

//------------------------------------------------------------------------------
ProcessGroupVEDA::ProcessGroupVEDA(int rank, int size, MPI_Comm pgComm) :
	Backend		(rank, size), stop_(false), pgComm_(pgComm),
	isTrace_	(tungl_is_active(TUNGL_LEVEL_DEBUG))
{
	if (pgComm_ == MPI_COMM_NULL) {
		TORCH_CHECK(false, "pgComm_ must not be MPI_COMM_NULL");
	}

	// Start the worker thread accepting MPI calls
	workerThread_ = std::thread(&ProcessGroupVEDA::runLoop, this);

	init();
}

//------------------------------------------------------------------------------
ProcessGroupVEDA::~ProcessGroupVEDA() {
	destroy();
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::destroy() {
	std::unique_lock<std::mutex> lock(pgMutex_);
	queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

	// Queue is empty, signal stop
	stop_ = true;

	// Release lock to allow threads to terminate
	lock.unlock();
	queueProduceCV_.notify_all();

	// Join the single worker thread
	workerThread_.join();
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::abort() {
	destroy();
	MPI_Abort(pgComm_, EXIT_FAILURE);
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::runLoop() {
	std::unique_lock<std::mutex> lock(pgMutex_);

	while (!stop_) {
		if (queue_.empty()) {
		queueProduceCV_.wait(lock);
		continue;
		}

		auto workTuple = std::move(queue_.front());

		queue_.pop_front();

		auto& workEntry = std::get<0>(workTuple);
		auto& work = std::get<1>(workTuple);

		lock.unlock();
		queueConsumeCV_.notify_one();

		try {
		workEntry->run(workEntry);
		work->finishWorkMPI();
		} catch (...) {
		work->finishWorkMPIError(std::current_exception());
		}

		lock.lock();
	}
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::enqueue(
    std::unique_ptr<WorkEntry> entry,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
	auto work =
		c10::make_intrusive<WorkMPI>(entry->dst, profilingTitle, inputTensors);
	std::unique_lock<std::mutex> lock(pgMutex_);
	queue_.emplace_back(std::move(entry), work);
	lock.unlock();
	queueProduceCV_.notify_one();
	return work;
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
	checkSingleTensor(tensors);
	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto data = (entry->src)[0];
			c10::DeviceGuard guard(data.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Bcast(
				deref(data.data_ptr()),
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				opts.rootRank,
				pgComm_));
		};
	auto entry =
		std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:broadcast",
		std::optional<std::vector<at::Tensor>>(tensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::allreduce(
	std::vector<at::Tensor>& tensors,
	const AllreduceOptions& opts) {
	checkSingleTensor(tensors);

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto data = (entry->src)[0];
			c10::DeviceGuard guard(data.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Allreduce(
				MPI_IN_PLACE,
				deref(data.data_ptr()),
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				mpiOp.at(opts.reduceOp),
				pgComm_));
		};
	auto entry =
		std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:all_reduce",
		std::optional<std::vector<at::Tensor>>(tensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::allreduce_coalesced(
	std::vector<at::Tensor>& tensors,
	const AllreduceCoalescedOptions& opts) {
	TORCH_CHECK(false, "allreduce_coalesced is currently not supported with VEDA");
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::reduce(
	std::vector<at::Tensor>& tensors,
	const ReduceOptions& opts) {
	checkSingleTensor(tensors);

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto data = (entry->src)[0];
			auto dataPtr = deref((entry->src)[0].data_ptr());
			void* sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
			void* recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

			c10::DeviceGuard guard(data.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Reduce(
				sendbuf,
				recvbuf,
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				mpiOp.at(opts.reduceOp),
				opts.rootRank,
				pgComm_));
		};
	auto entry =
		std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:reduce",
		std::optional<std::vector<at::Tensor>>(tensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::allgather(
	std::vector<std::vector<at::Tensor>>& outputTensors,
	std::vector<at::Tensor>& inputTensors,
	const AllgatherOptions& opts) {
	checkSingleTensor(inputTensors);
	if (outputTensors.size() != 1) {
		TORCH_CHECK(
			false,
			"MPI process group only supports a single "
			"tensor op");
	}
	if (static_cast<size_t>(size_) != outputTensors[0].size()) {
		TORCH_CHECK(
			false,
			"All gather: number of output tensors should equal "
			"to the world size");
	}

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc = [this](std::unique_ptr<WorkEntry>& entry) {
		auto start	= traceStart();
		int rank	= 0;
		
		for(auto& out : entry->dst) {
			MPI_CHECK(MPI_Bcast(
				deref(out.data_ptr()),
				out.numel(),
				mpiDatatype.at(out.scalar_type()),
				rank++,
				pgComm_));
		}
		
		traceEnd(__LINE__, "allgather", start, entry->dst);
	};

	auto entry = std::make_unique<WorkEntry>(
		&inputTensors, &outputTensors[0], std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:all_gather",
		std::optional<std::vector<at::Tensor>>(inputTensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::allgather_coalesced(
		std::vector<std::vector<at::Tensor>>& /* unused */,
		std::vector<at::Tensor>& /* unused */,
		const AllgatherOptions& /* unused */) {
	TORCH_CHECK(false, "ProcessGroupVEDA does not support allgather_coalesced");
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::gather(
		std::vector<std::vector<at::Tensor>>& outputTensors,
		std::vector<at::Tensor>& inputTensors,
		const GatherOptions& opts) {
	checkSingleTensor(inputTensors);

	if (rank_ != opts.rootRank) {
		if (!outputTensors.empty()) {
		TORCH_CHECK(
			false,
			"Gather: number of output tensors should be 0 "
			"for non-root");
		}
	} else {
		if (outputTensors.size() != 1) {
		TORCH_CHECK(false, "Gather: multi-GPU collective is not supported");
		}
		if (static_cast<size_t>(size_) != outputTensors[0].size()) {
		TORCH_CHECK(
			false,
			"Gather: number of output tensors should equal "
			"to the world size");
		}
		checkSameSizeAndType(inputTensors[0], outputTensors[0]);
	}

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto data = (entry->src)[0];
			void* recvbuf = nullptr;
			at::Tensor flatOutputTensor;

			std::vector<at::Tensor> dstdata = entry->dst;
			if (rank_ == opts.rootRank) {
			flatOutputTensor = newLikeFlat(dstdata);
			recvbuf = deref(flatOutputTensor.data_ptr());
			}

			c10::DeviceGuard guard(data.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Gather(
				deref(data.data_ptr()),
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				recvbuf,
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				opts.rootRank,
				pgComm_));

			if (rank_ == opts.rootRank) {
			const std::vector<at::Tensor>& outputDataVec = entry->dst;
			// copy the flattened output tensors to the outputs
			for (const auto i : c10::irange(outputDataVec.size())) {
				outputDataVec.at(i).copy_(
					flatOutputTensor[static_cast<int64_t>(i)]);
			}
			}
		};

	if (rank_ == opts.rootRank) {
		auto entry = std::make_unique<WorkEntry>(
			&inputTensors, &outputTensors[0], std::move(runFunc));
		return enqueue(
			std::move(entry),
			"veda:gather",
			std::optional<std::vector<at::Tensor>>(inputTensors));
	} else {
		auto entry =
			std::make_unique<WorkEntry>(&inputTensors, nullptr, std::move(runFunc));
		return enqueue(
			std::move(entry),
			"veda:gather",
			std::optional<std::vector<at::Tensor>>(inputTensors));
	}
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::scatter(
		std::vector<at::Tensor>& outputTensors,
		std::vector<std::vector<at::Tensor>>& inputTensors,
		const ScatterOptions& opts) {
	checkSingleTensor(outputTensors);

	if (rank_ != opts.rootRank) {
		if (!inputTensors.empty()) {
		TORCH_CHECK(
			false,
			"Scatter: number of input tensors should be 0 "
			"for non-root");
		}
	} else {
		if (inputTensors.size() != 1) {
		TORCH_CHECK(false, "Scatter: multi-GPU collective is not supported");
		}
		if (static_cast<size_t>(size_) != inputTensors[0].size()) {
		TORCH_CHECK(
			false,
			"Scatter: number of input tensors should equal "
			"to the world size");
		}
		checkSameSizeAndType(outputTensors[0], inputTensors[0]);
	}

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto data = (entry->dst)[0];
			void* sendbuf = nullptr;
			at::Tensor flatInputTensor;

			if (rank_ == opts.rootRank) {
				std::vector<at::Tensor>& inputDataVec = entry->src;
				flatInputTensor = newLikeFlat(inputDataVec);
				sendbuf = deref(flatInputTensor.data_ptr());

				// copy the input tensors to the flatten large send buffer
				for (const auto i : c10::irange(inputDataVec.size())) {
					flatInputTensor[static_cast<int64_t>(i)].copy_(inputDataVec.at(i));
				}
			}

			c10::DeviceGuard guard(data.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Scatter(
				sendbuf,
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				deref(data.data_ptr()),
				data.numel(),
				mpiDatatype.at(data.scalar_type()),
				opts.rootRank,
				pgComm_));
		};

	if (rank_ == opts.rootRank) {
		auto entry = std::make_unique<WorkEntry>(
			&inputTensors[0], &outputTensors, std::move(runFunc));
		return enqueue(
			std::move(entry),
			"veda:scatter",
			!inputTensors.empty()
				? std::optional<std::vector<at::Tensor>>(inputTensors[0])
				: std::nullopt);
	} else {
		auto entry = std::make_unique<WorkEntry>(
			nullptr, &outputTensors, std::move(runFunc));
		return enqueue(
			std::move(entry),
			"veda:scatter",
			!inputTensors.empty()
				? std::optional<std::vector<at::Tensor>>(inputTensors[0])
				: std::nullopt);
	}
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::reduce_scatter(
		std::vector<at::Tensor>& outputTensors,
		std::vector<std::vector<at::Tensor>>& inputTensors,
		const ReduceScatterOptions& opts) {
	checkSingleTensor(outputTensors);
	if (inputTensors.size() != 1) {
		TORCH_CHECK(
			false,
			"MPI process group only supports a single "
			"tensor op");
	}
	if (static_cast<size_t>(size_) != inputTensors[0].size()) {
		TORCH_CHECK(
			false,
			"Reduce scatter: number of input tensors should equal "
			"to the world size");
	}
	checkSameSizeAndType(outputTensors[0], inputTensors[0]);

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto data = (entry->dst)[0];
			auto flatInputTensor = newLikeFlat(entry->src);
			for (const auto i : c10::irange(entry->src.size())) {
				flatInputTensor[static_cast<int64_t>(i)].copy_(entry->src[i]);
			}
			int recvcount = flatInputTensor.numel() / size_;

			c10::DeviceGuard guard(data.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Reduce_scatter_block(
				deref(flatInputTensor.data_ptr()),
				deref(data.data_ptr()),
				recvcount,
				mpiDatatype.at(data.scalar_type()),
				mpiOp.at(opts.reduceOp),
				pgComm_));
		};

	auto entry = std::make_unique<WorkEntry>(
		&inputTensors[0], &outputTensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:reduce_scatter",
		std::optional<std::vector<at::Tensor>>(inputTensors[0]));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::alltoall_base(
		at::Tensor& outputTensor,
		at::Tensor& inputTensor,
		std::vector<int64_t>& outputSplitSizes,
		std::vector<int64_t>& inputSplitSizes,
		const AllToAllOptions& opts) {
	checkSingleTensorHelper(inputTensor);
	checkSingleTensorHelper(outputTensor);

	if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
		// We can use alltoall
		TORCH_CHECK(
			outputTensor.numel() == inputTensor.numel() &&
				outputTensor.scalar_type() == inputTensor.scalar_type(), // MODIFED: fixed deprecation warning
			"Tensors are not equal in size or data type");
		TORCH_CHECK(
			outputTensor.size(0) % size_ == 0,
			"Tensor's dim 0 does not divide equally across group size");

		std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
			[this](std::unique_ptr<WorkEntry>& entry) {
			auto start = traceStart();
			auto srcdata = (entry->src)[0];
			auto dstdata = (entry->dst)[0];
			c10::DeviceGuard guard(srcdata.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Alltoall(
				deref(srcdata.data_ptr()),
				srcdata.numel() / size_,
				mpiDatatype.at(srcdata.scalar_type()),
				deref(dstdata.data_ptr()),
				dstdata.numel() / size_,
				mpiDatatype.at(dstdata.scalar_type()),
				pgComm_));
			traceEnd(__LINE__, "alltoall_base", start, entry->src);
		};
		std::vector<at::Tensor> inputTensors = {inputTensor};
		std::vector<at::Tensor> outputTensors = {outputTensor};
		auto entry = std::make_unique<WorkEntry>(
			&inputTensors, &outputTensors, std::move(runFunc));
		return enqueue(
			std::move(entry),
			"veda:all_to_all",
			std::optional<std::vector<at::Tensor>>(inputTensors));
	} else {
		// Need alltoallv
		c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
		c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
		std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
			[this, inputSplitSizes, outputSplitSizes](
				std::unique_ptr<WorkEntry>& entry) {
			auto start = traceStart();
			auto srcdata = (entry->src)[0];
			auto dstdata = (entry->dst)[0];
			std::vector<int> send_lengths(size_);
			std::vector<int> recv_lengths(size_);
			std::vector<int> send_offsets(size_);
			std::vector<int> recv_offsets(size_);
			c10d::computeLengthsAndOffsets(
				inputSplitSizes, srcdata, &send_lengths, &send_offsets);
			c10d::computeLengthsAndOffsets(
				outputSplitSizes, dstdata, &recv_lengths, &recv_offsets);
			c10::DeviceGuard guard(srcdata.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Alltoallv(
				deref(srcdata.data_ptr()),
				send_lengths.data(),
				send_offsets.data(),
				mpiDatatype.at(srcdata.scalar_type()),
				deref(dstdata.data_ptr()),
				recv_lengths.data(),
				recv_offsets.data(),
				mpiDatatype.at(dstdata.scalar_type()),
				pgComm_));
			traceEnd(__LINE__, "alltoall_base", start, entry->src);
			};
		std::vector<at::Tensor> inputTensors = {inputTensor};
		std::vector<at::Tensor> outputTensors = {outputTensor};
		auto entry = std::make_unique<WorkEntry>(
			&inputTensors, &outputTensors, std::move(runFunc));
		return enqueue(
			std::move(entry),
			"veda:all_to_all",
			std::optional<std::vector<at::Tensor>>(inputTensors));
	}
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::alltoall(
		std::vector<at::Tensor>& outputTensors,
		std::vector<at::Tensor>& inputTensors,
		const AllToAllOptions& opts) {
	TORCH_CHECK(
		inputTensors.size() == static_cast<size_t>(size_),
		"Number of input tensors are not equal to group size");
	TORCH_CHECK(
		outputTensors.size() == static_cast<size_t>(size_),
		"Number of output tensors are not equal to group size");
	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[this](std::unique_ptr<WorkEntry>& entry) {
			auto start = traceStart();
			std::vector<int> send_lengths(size_);
			std::vector<int> recv_lengths(size_);
			std::vector<int> send_offsets(size_);
			std::vector<int> recv_offsets(size_);
			auto srcdata = entry->src;
			auto dstdata = entry->dst;
			auto src_len = c10d::computeLengthsAndOffsets(
				srcdata, &send_lengths, &send_offsets);
			auto dst_len = c10d::computeLengthsAndOffsets(
				dstdata, &recv_lengths, &recv_offsets);
			std::vector<int64_t> send_lengthsL(
				send_lengths.begin(), send_lengths.end());
			std::vector<int64_t> recv_lengthsL(
				recv_lengths.begin(), recv_lengths.end());
			at::Tensor srcFlatData =
				at::empty({static_cast<int64_t>(src_len)}, srcdata[0].options());
			at::Tensor dstFlatData =
				at::empty({static_cast<int64_t>(dst_len)}, dstdata[0].options());
			auto srcFlatDataSplits =
				srcFlatData.split_with_sizes(c10::IntArrayRef(send_lengthsL), 0);
			for (const auto i : c10::irange(size_)) {
			srcFlatDataSplits[i].copy_(srcdata[i].view({-1}));
			}
			c10::DeviceGuard guard1(srcdata[0].device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Alltoallv(
				deref(srcFlatData.data_ptr()),
				send_lengths.data(),
				send_offsets.data(),
				mpiDatatype.at(srcdata[0].scalar_type()),
				deref(dstFlatData.data_ptr()),
				recv_lengths.data(),
				recv_offsets.data(),
				mpiDatatype.at(dstdata[0].scalar_type()),
				pgComm_));

			auto dstFlatDataSplits =
				dstFlatData.split_with_sizes(c10::IntArrayRef(recv_lengthsL), 0);
			for (const auto i : c10::irange(size_)) {
			dstdata[i].view({-1}).copy_(dstFlatDataSplits[i]);
			}
			traceEnd(__LINE__, "alltoall", start, srcdata);
		};
	auto entry = std::make_unique<WorkEntry>(
		&inputTensors, &outputTensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:all_to_all",
		std::optional<std::vector<at::Tensor>>(inputTensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::send(
		std::vector<at::Tensor>& tensors,
		int dstRank,
		int tag) {
	checkSingleTensor(tensors);

	auto& tensor = tensors[0];
	MPI_Request request = MPI_REQUEST_NULL;

	{
		c10::DeviceGuard guard(tensor.device());
		std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
		MPI_CHECK(MPI_Isend(
			deref(tensor.data_ptr()),
			tensor.numel(),
			mpiDatatype.at(tensor.scalar_type()),
			dstRank,
			tag,
			pgComm_,
			&request));
	}

	return c10::make_intrusive<AsyncWork>(
		request,
		std::vector<at::Tensor>(),
		"veda:send",
		std::optional<std::vector<at::Tensor>>(tensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::recv(
		std::vector<at::Tensor>& tensors,
		int srcRank,
		int tag) {
	checkSingleTensor(tensors);

	auto& tensor = tensors[0];
	MPI_Request request = MPI_REQUEST_NULL;

	{
		c10::DeviceGuard guard(tensor.device());
		std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
		MPI_CHECK(MPI_Irecv(
			deref(tensor.data_ptr()),
			tensor.numel(),
			mpiDatatype.at(tensor.scalar_type()),
			srcRank,
			tag,
			pgComm_,
			&request));
	}

	return c10::make_intrusive<AsyncWork>(
		request,
		tensors,
		"veda:recv",
		std::optional<std::vector<at::Tensor>>(tensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::recvAnysource(
		std::vector<at::Tensor>& tensors,
		int tag) {
	checkSingleTensor(tensors);

	auto& tensor = tensors[0];
	MPI_Request request = MPI_REQUEST_NULL;

	{
		c10::DeviceGuard guard(tensor.device());
		std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
		MPI_CHECK(MPI_Irecv(
			deref(tensor.data_ptr()),
			tensor.numel(),
			mpiDatatype.at(tensor.scalar_type()),
			MPI_ANY_SOURCE,
			tag,
			pgComm_,
			&request));
	}

	return c10::make_intrusive<AsyncWork>(
		request,
		tensors,
		"veda:recvAnySource",
		std::optional<std::vector<at::Tensor>>(tensors));
	}

	c10::intrusive_ptr<Work> ProcessGroupVEDA::barrier(const BarrierOptions& opts) {
	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[this](std::unique_ptr<WorkEntry>& entry) {
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Barrier(pgComm_));
		};
	auto entry =
		std::make_unique<WorkEntry>(nullptr, nullptr, std::move(runFunc));
	return enqueue(std::move(entry), "veda:barrier", std::nullopt);
	}

	c10::intrusive_ptr<Work> ProcessGroupVEDA::_allgather_base(
		at::Tensor& outputTensor,
		at::Tensor& inputTensor,
		const AllgatherOptions& opts) {
	TORCH_CHECK(
		outputTensor.numel() == inputTensor.numel() * size_,
		"All gather: output tensor size must be equal to input tensor size times the world size");

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[this](std::unique_ptr<WorkEntry>& entry) {
			auto dstdata = (entry->dst)[0];
			auto srcdata = (entry->src)[0];
			c10::DeviceGuard guard(srcdata.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Allgather(
				deref(srcdata.data_ptr()),
				srcdata.numel(),
				mpiDatatype.at(srcdata.scalar_type()),
				deref(dstdata.data_ptr()),
				srcdata.numel(),
				mpiDatatype.at(dstdata.scalar_type()),
				pgComm_));
		};

	auto inputTensors = std::vector<at::Tensor>({inputTensor});
	auto outputTensors = std::vector<at::Tensor>({outputTensor});
	auto entry = std::make_unique<WorkEntry>(
		&inputTensors, &outputTensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:_allgather_base",
		std::optional<std::vector<at::Tensor>>(inputTensors));
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::_reduce_scatter_base(
		at::Tensor& outputTensor,
		at::Tensor& inputTensor,
		const ReduceScatterOptions& opts) {
	TORCH_CHECK(
		outputTensor.numel() * size_ == inputTensor.numel(),
		"Reduce scatter: input tensor size must be equal to output tensor size times the world size");

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
		[opts, this](std::unique_ptr<WorkEntry>& entry) {
			auto dstdata = (entry->dst)[0];
			auto srcdata = (entry->src)[0];
			c10::DeviceGuard guard(srcdata.device());
			std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
			MPI_CHECK(MPI_Reduce_scatter_block(
				deref(srcdata.data_ptr()),
				deref(dstdata.data_ptr()),
				dstdata.numel(),
				mpiDatatype.at(srcdata.scalar_type()),
				mpiOp.at(opts.reduceOp),
				pgComm_));
		};

	auto inputTensors = std::vector<at::Tensor>({inputTensor});
	auto outputTensors = std::vector<at::Tensor>({outputTensor});
	auto entry = std::make_unique<WorkEntry>(
		&inputTensors, &outputTensors, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"veda:_reduce_scatter_base",
		std::optional<std::vector<at::Tensor>>(inputTensors));
}

//------------------------------------------------------------------------------
uint64_t ProcessGroupVEDA::traceStart(void) const {
	if(isTrace_) {
		CVEDA(vedaStreamSynchronize(0));
		return timer();
	}
	return 0;
}

//------------------------------------------------------------------------------
void ProcessGroupVEDA::traceEnd(const int line, const char* func, const uint64_t start, const std::vector<at::Tensor>& tensors) const {
	if(isTrace_) {
		const auto ms = (timer() - start)/1000000.0;
		size_t mb_ = 0;
		for(auto& t : tensors)
			mb_ += t.numel() * t.element_size();
		const auto mb = mb_ / 1024.0 / 1024.0;
		tungl_log(TUNGL_LEVEL_DEBUG, L_MODULE, __FILE__, line, "%s[%i/%i]: %f MB, %f ms, %f GB/s ", func, getRank(), getSize(), mb, ms, mb / ms / 1.024);
	}
}

//------------------------------------------------------------------------------
void* ProcessGroupVEDA::deref(void* vptr) const {
	VEDAhmemptr hmem = 0;
	CVEDA(vedaMemToHMEM(&hmem, (VEDAdeviceptr)vptr));
	return hmem;
}

//------------------------------------------------------------------------------
c10::intrusive_ptr<Work> ProcessGroupVEDA::allgather_into_tensor_coalesced(
	std::vector<at::Tensor>& outputs,
	std::vector<at::Tensor>& inputs,
	const AllgatherOptions& opts
) {
	TORCH_CHECK_EQ(outputs.size(), inputs.size());

	std::function<void(std::unique_ptr<WorkEntry>&)> runFunc = [this](std::unique_ptr<WorkEntry>& entry) {
		auto start		= traceStart();
		auto oit		= entry->dst.begin();
		auto iit		= entry->src.begin();
		const int cnt	= getSize();
		for(; oit != entry->dst.end(); oit++, iit++) {
			auto& out	= *oit;
			auto& in	= *iit;

			const auto dtype	= mpiDatatype.at(out.scalar_type());
			const auto numel	= in.numel();
			const size_t bytes	= numel * in.element_size();
			CVEDA(vedaMemcpyDtoDAsync(((VEDAdeviceptr)out.data_ptr()) + bytes * getRank(), (VEDAdeviceptr)in.data_ptr(), bytes, 0));

			auto ptr = (char*)deref(out.data_ptr());
			for(int rank = 0; rank < cnt; rank++, ptr += bytes) {
				MPI_CHECK(MPI_Bcast(ptr, numel, dtype, rank, pgComm_));
			}

			CVEDA(vedaStreamSynchronize(0));
		}
		traceEnd(__LINE__, "allgather_into_tensor_coalesced", start, entry->dst);
	};

	auto entry = std::make_unique<WorkEntry>(&inputs, &outputs, std::move(runFunc));
	return enqueue(
		std::move(entry),
		"allgather_into_tensor_coalesced",
		std::optional<std::vector<at::Tensor>>(inputs));
}

//------------------------------------------------------------------------------

} // namespace c10d

