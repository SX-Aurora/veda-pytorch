/**
 * Code taken and adjusted from: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/init.cpp
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

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>

extern "C" PyObject* PyInit__C(void) {
	#define PY_CHECK(X) if(!(X)) throw python_error()
	auto module_    = PyImport_ImportModule("torch._C._distributed_c10d");	PY_CHECK(module_);
    auto dict       = PyModule_GetDict(module_);							PY_CHECK(dict);
	auto backend_   = PyDict_GetItemString(dict, "Backend");			    PY_CHECK(backend_);
	auto module     = py::handle(module_).cast<py::module>();
	auto backend    = py::handle(backend_).cast<py::class_<::c10d::Backend, c10::intrusive_ptr<::c10d::Backend>>>();
    #undef PY_CHECK

    auto processGroupVEDA =
        py::class_<::c10d::ProcessGroupVEDA, c10::intrusive_ptr<::c10d::ProcessGroupVEDA>>(module, "ProcessGroupVEDA", backend);
	processGroupVEDA.def_static("create",
		[](const ::c10d::DistributedBackendOptions& options) {
			std::vector<int> ranks;
			ranks.reserve(options.global_ranks_in_group.size());
			for(auto rank : options.global_ranks_in_group)
				ranks.emplace_back(rank);
		    return ::c10d::ProcessGroupVEDA::createProcessGroupVEDA(ranks);
		},
		py::call_guard<py::gil_scoped_release>());
	
    static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL};
    return PyModule_Create(&module_def);
}