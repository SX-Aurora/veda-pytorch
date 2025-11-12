/**
 * Code taken and adjusted from: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/ProcessGroupVEDA.hpp
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

#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <mpi.h>

namespace c10d {

constexpr const char* MPI_BACKEND_NAME = "veda";

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
	explicit WorkEntry(
		std::vector<at::Tensor>* srcPtr,
		std::vector<at::Tensor>* dstPtr,
		std::function<void(std::unique_ptr<WorkEntry>&)> run)
		: dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
		if (srcPtr) {
			src = *srcPtr;
		}
	}

	// Not copyable
	WorkEntry(const WorkEntry&) = delete;
	// Not copy assignable
	WorkEntry& operator=(const WorkEntry&) = delete;

	// For input and output tensors (in-place), we will always use src
	std::vector<at::Tensor> src;

	// Copy of user provided outputs.
	const std::vector<at::Tensor> dst;

	// src rank returned, for recv only
	int* srcRank = nullptr;
	std::function<void(std::unique_ptr<WorkEntry>&)> run;
};

// ProcessGroupVEDA implements MPI bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupVEDA requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupVEDA will only support a single process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupVEDA, it requires your MPI
// implementation to have a thread support value of MPI_THREAD_MULTIPLE, that
// is, multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupVEDA only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupVEDA will automatically detect this support.
class TORCH_API ProcessGroupVEDA : public Backend {
public:
	class WorkMPI : public Work {
	public:
		explicit WorkMPI(
			std::vector<at::Tensor> outputTensors,
			const char* profilingTitle = nullptr,
			const std::optional<std::vector<at::Tensor>>& inputTensors =
				std::nullopt)
			: Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
			outputTensors_(std::move(outputTensors)),
			future_(c10::make_intrusive<at::ivalue::Future>(
				c10::ListType::create(c10::TensorType::get()))) {}

		std::vector<at::Tensor> result() override;

		c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

	protected:
		friend class ProcessGroupVEDA;

	private:
		void finishWorkMPI();
		void finishWorkMPIError(const std::exception_ptr& eptr);

		std::vector<at::Tensor> outputTensors_;
		c10::intrusive_ptr<at::ivalue::Future> future_;
	};

	class AsyncWork : public Work {
	public:
		AsyncWork(
			MPI_Request request,
			std::vector<at::Tensor> outputTensors,
			const char* profilingTitle = nullptr,
			const std::optional<std::vector<at::Tensor>>& inputTensors =
				std::nullopt);

		~AsyncWork() override;

		bool isCompleted() override;

		bool isSuccess() const override;

		int sourceRank() const override;

		bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

		void abort() override;

		std::vector<at::Tensor> result() override;

		protected:
		void populateException();

	private:
		const std::vector<at::Tensor> outputTensors_;
		MPI_Request request_;
		MPI_Status status_{};
	};

private:
	const bool isTrace_;
	uint64_t	traceStart	(void) const;
	void		traceEnd	(const int line, const char* func, const uint64_t start, const std::vector<at::Tensor>& ref) const;

public:
	// Constructor will spawn up the worker thread loop
	explicit ProcessGroupVEDA(int rank, int size, MPI_Comm pgComm);

	~ProcessGroupVEDA() override;

	// Abort the MPI program, needs to be called when exception is detected
#if TORCH_VERSION_ >= 20700
	void abort() override;
#else
	void abort();
#endif

	const std::string getBackendName() const override {
		return std::string(MPI_BACKEND_NAME);
	}

	c10::intrusive_ptr<Work> broadcast(
		std::vector<at::Tensor>& data,
		const BroadcastOptions& opts = BroadcastOptions()) override;

	c10::intrusive_ptr<Work> allreduce(
		std::vector<at::Tensor>& tensors,
		const AllreduceOptions& opts = AllreduceOptions()) override;

	c10::intrusive_ptr<Work> allreduce_coalesced(
		std::vector<at::Tensor>& tensors,
		const AllreduceCoalescedOptions& opts =
			AllreduceCoalescedOptions()) override;

	c10::intrusive_ptr<Work> reduce(
		std::vector<at::Tensor>& tensors,
		const ReduceOptions& opts = ReduceOptions()) override;

	c10::intrusive_ptr<Work> allgather(
		std::vector<std::vector<at::Tensor>>& outputTensors,
		std::vector<at::Tensor>& inputTensors,
		const AllgatherOptions& opts = AllgatherOptions()) override;

	c10::intrusive_ptr<Work> _allgather_base(
		at::Tensor& outputbuffer,
		at::Tensor& inputbuffer,
		const AllgatherOptions& opts = AllgatherOptions()) override;

	c10::intrusive_ptr<Work> allgather_coalesced(
		std::vector<std::vector<at::Tensor>>& outputTensorLists,
		std::vector<at::Tensor>& inputTensors,
		const AllgatherOptions& opts = AllgatherOptions()) override;

	c10::intrusive_ptr<Work> gather(
		std::vector<std::vector<at::Tensor>>& outputTensors,
		std::vector<at::Tensor>& inputTensors,
		const GatherOptions& opts = GatherOptions()) override;

	c10::intrusive_ptr<Work> scatter(
		std::vector<at::Tensor>& outputTensors,
		std::vector<std::vector<at::Tensor>>& inputTensors,
		const ScatterOptions& opts = ScatterOptions()) override;

	c10::intrusive_ptr<Work> reduce_scatter(
		std::vector<at::Tensor>& outputTensors,
		std::vector<std::vector<at::Tensor>>& inputTensors,
		const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

	c10::intrusive_ptr<Work> _reduce_scatter_base(
		at::Tensor& outputTensor,
		at::Tensor& inputTensor,
		const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

	c10::intrusive_ptr<Work> alltoall_base(
		at::Tensor& outputTensor,
		at::Tensor& inputTensor,
		std::vector<int64_t>& outputSplitSizes,
		std::vector<int64_t>& inputSplitSizes,
		const AllToAllOptions& opts = AllToAllOptions()) override;

	c10::intrusive_ptr<Work> alltoall(
		std::vector<at::Tensor>& outputTensors,
		std::vector<at::Tensor>& inputTensors,
		const AllToAllOptions& opts = AllToAllOptions()) override;

	c10::intrusive_ptr<Work> send(
		std::vector<at::Tensor>& tensors,
		int dstRank,
		int tag) override;

	c10::intrusive_ptr<Work> recv(
		std::vector<at::Tensor>& tensors,
		int srcRank,
		int tag) override;

	c10::intrusive_ptr<Work> recvAnysource(
		std::vector<at::Tensor>& tensor,
		int tag) override;

	c10::intrusive_ptr<Work> barrier(
		const BarrierOptions& opts = BarrierOptions()) override;

		// MODIFIED
		c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
			std::vector<at::Tensor>& outputs,
			std::vector<at::Tensor>& inputs,
			const AllgatherOptions& opts = AllgatherOptions()) override;

		void* deref(void* vptr) const;
		// /MODIFIED

	// Creating a new ProcessGroupVEDA, will initialize MPI if not initialized
	static c10::intrusive_ptr<ProcessGroupVEDA> createProcessGroupVEDA(
		std::vector<int> ranks = {});

protected:
	using WorkType =
		std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkMPI>>;
	// Worker thread loop
	void runLoop();
	// Helper function that is called by the destructor
	void destroy();

	c10::intrusive_ptr<Work> enqueue(
		std::unique_ptr<WorkEntry> entry,
		const char* profilingTitle = nullptr,
		const std::optional<std::vector<at::Tensor>>& inputTensors =
			std::nullopt);

	bool stop_;

	std::mutex pgMutex_;
	std::thread workerThread_;

	std::deque<WorkType> queue_;
	std::condition_variable queueProduceCV_;
	std::condition_variable queueConsumeCV_;

	// Global states
	static void initMPIOnce();
	static void mpiExit();

	static std::mutex pgGlobalMutex_;
	static int mpiThreadSupport_;

	MPI_Comm pgComm_;
};

} // namespace c10d