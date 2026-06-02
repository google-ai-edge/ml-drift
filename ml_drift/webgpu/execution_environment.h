#ifndef ML_DRIFT_WEBGPU_EXECUTION_ENVIRONMENT_H_
#define ML_DRIFT_WEBGPU_EXECUTION_ENVIRONMENT_H_

#include "ml_drift/webgpu/environment.h"

namespace ml_drift {
namespace webgpu {

// TODO(cl/432271593): Remove this alias once all users are migrated to
// Environment.
using ExecutionEnvironment = ml_drift::webgpu::Environment;

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_EXECUTION_ENVIRONMENT_H_
