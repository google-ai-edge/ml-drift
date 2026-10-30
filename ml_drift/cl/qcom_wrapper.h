// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ML_DRIFT_CL_QCOM_WRAPPER_H_
#define ML_DRIFT_CL_QCOM_WRAPPER_H_

#include <cstddef>

#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace ml_drift {
namespace cl {

/*********************************
 * cl_qcom_accelerated_image_ops extension *
 *********************************/
#define CL_MEM_OBJECT_WEIGHT_IMAGE_QCOM 0x4110
#define CL_DEVICE_HOF_MAX_NUM_PHASES_QCOM 0x4111
#define CL_DEVICE_HOF_MAX_FILTER_SIZE_X_QCOM 0x4112
#define CL_DEVICE_HOF_MAX_FILTER_SIZE_Y_QCOM 0x4113
#define CL_DEVICE_BLOCK_MATCHING_MAX_REGION_SIZE_X_QCOM 0x4114
#define CL_DEVICE_BLOCK_MATCHING_MAX_REGION_SIZE_Y_QCOM 0x4115

// Extended flag for specifying weight image type
#define CL_WEIGHT_IMAGE_SEPARABLE_QCOM (1 << 0)

// Box Filter
typedef struct _cl_box_filter_size_qcom {
  // Width of box filter on X direction.
  float box_filter_width;

  // Height of box filter on Y direction.
  float box_filter_height;
} cl_box_filter_size_qcom;

// HOF Weight Image Desc
typedef struct _cl_weight_desc_qcom {
  /** Coordinate of the "center" point of the weight image,
      based on the weight image's top-left corner as the origin. */
  size_t center_coord_x;
  size_t center_coord_y;
  cl_bitfield flags;
} cl_weight_desc_qcom;

typedef struct _cl_weight_image_desc_qcom {
  cl_image_desc image_desc;
  cl_weight_desc_qcom weight_desc;
} cl_weight_image_desc_qcom;

/*************************************
 * cl_recordable_command_queue extension *
 *************************************/

/** Accepted by clGetDeviceInfo */
#define CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE 0x41DE

/** Flag to enable recordable command queues */
#define CL_QUEUE_RECORDABLE_QCOM (1u << 30u)

typedef struct _cl_recording_qcom *cl_recording_qcom;

/** Array element struct used to set kernel arguments */
typedef struct _cl_array_arg_qcom {
  cl_uint dispatch_index;
  cl_uint arg_index;
  size_t arg_size;
  const void *arg_value;
} cl_array_arg_qcom;

typedef struct _cl_array_kernel_exec_info_qcom {
  cl_uint dispatch_index;
  cl_kernel_exec_info param_name;
  size_t param_value_size;
  const void *param_value;
} cl_array_kernel_exec_info_qcom;

/** Used to update a local or global workgroup.  workgroup_size * is used in the
   same manner as the correponding argument in clEnqueueNDRangeKernel */
typedef struct _cl_workgroup_qcom {
  cl_uint dispatch_index;
  const size_t *workgroup_size;
} cl_workgroup_qcom;

typedef struct _cl_offset_qcom {
  cl_uint dispatch_index;
  size_t offsets[3];
} cl_offset_qcom;

typedef cl_recording_qcom (*PFN_clNewRecordingQCOM)(
    cl_command_queue command_queue, cl_int *errcode_ret);

typedef cl_int (*PFN_clEndRecordingQCOM)(cl_recording_qcom recording);

typedef cl_int (*PFN_clEnqueueRecordingQCOM)(
    cl_command_queue command_queue, cl_recording_qcom recording,
    size_t num_args, const cl_array_arg_qcom *arg_array,
    size_t num_global_offsets, const cl_offset_qcom *global_offset_array,
    size_t num_global_workgroups,
    const cl_workgroup_qcom *global_workgroup_array,
    size_t num_local_workgroups,
    const cl_workgroup_qcom *local_workgroups_array,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);

typedef cl_int (*PFN_clReleaseRecordingQCOM)(cl_recording_qcom recording);

extern PFN_clNewRecordingQCOM clNewRecordingQCOM;
extern PFN_clEndRecordingQCOM clEndRecordingQCOM;
extern PFN_clEnqueueRecordingQCOM clEnqueueRecordingQCOM;
extern PFN_clReleaseRecordingQCOM clReleaseRecordingQCOM;

/*************************************
 * cl_qcom_perf_hint extension *
 *************************************/

typedef cl_uint cl_perf_hint;

#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2

/*cl_perf_hint*/
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5

typedef cl_int (*PFN_clSetPerfHintQCOM)(cl_context context,
                                        cl_perf_hint perf_hint);

extern PFN_clSetPerfHintQCOM clSetPerfHintQCOM;

/*************************************
 * cl_qcom_priority_hint extension *
 *************************************/
// This extension lets the application specify the context priority.

#define CL_PRIORITY_HINT_NONE_QCOM 0
typedef cl_uint cl_priority_hint;

#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9

/*cl_priority_hint*/
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

/*************************************
 * Extension function loading *
 *************************************/

#define LoadQcomExtensionFunctions()    \
  LoadFunction(clNewRecordingQCOM);     \
  LoadFunction(clEndRecordingQCOM);     \
  LoadFunction(clEnqueueRecordingQCOM); \
  LoadFunction(clReleaseRecordingQCOM); \
  LoadFunction(clSetPerfHintQCOM);

#define DEFINE_QCOM_FUNCTION_PTRS                    \
  PFN_clNewRecordingQCOM clNewRecordingQCOM;         \
  PFN_clEndRecordingQCOM clEndRecordingQCOM;         \
  PFN_clEnqueueRecordingQCOM clEnqueueRecordingQCOM; \
  PFN_clReleaseRecordingQCOM clReleaseRecordingQCOM; \
  PFN_clSetPerfHintQCOM clSetPerfHintQCOM;

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_QCOM_WRAPPER_H_
