// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"

#include <cassert>
#include <memory>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/common/status.h"
#include "core/common/safeint.h"

using onnxruntime::common::Status;

struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 28196)
#pragma warning(disable : 6387)
#endif

namespace {
inline OrtStatus* NewStatus(size_t clen) {
  auto* buf = new (std::nothrow) uint8_t[sizeof(OrtStatus) + clen];
  if (buf == nullptr) return nullptr;  // OOM. What we can do here? abort()?
  return new (buf) OrtStatus;
}

inline void DeleteStatus(OrtStatus* ort_status) {
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
  delete[] reinterpret_cast<uint8_t*>(ort_status);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
}
}  // namespace

// Even we say it may not return NULL, indeed it may.
_Check_return_ _Ret_notnull_ OrtStatus* ORT_API_CALL OrtApis::CreateStatus(OrtErrorCode code,
                                                                           _In_z_ const char* msg) NO_EXCEPTION {
  assert(!(code == 0 && msg != nullptr));
  SafeInt<size_t> clen(nullptr == msg ? 0 : strnlen(msg, onnxruntime::kMaxStrLen));
  OrtStatus* p = NewStatus(clen);
  if (p == nullptr)
    return nullptr;
  p->code = code;
  memcpy(p->msg, msg, clen);
  p->msg[clen] = '\0';
  return p;
}

namespace onnxruntime {

namespace {

struct OrtStatusDeleter {
  void operator()(OrtStatus* p) const noexcept {
    if (p != nullptr) {
      DeleteStatus(p);
    }
  }
};

using UniqueOrtStatus = std::unique_ptr<OrtStatus, OrtStatusDeleter>;

}  // namespace

_Ret_notnull_ OrtStatus* ToOrtStatus(const Status& st) {
  if (st.IsOK())
    return nullptr;
  SafeInt<size_t> clen(st.ErrorMessage().length());
  OrtStatus* p = NewStatus(clen);
  if (p == nullptr)
    return nullptr;
  p->code = static_cast<OrtErrorCode>(st.Code());
  memcpy(p->msg, st.ErrorMessage().c_str(), clen);
  p->msg[clen] = '\0';
  return p;
}

Status ToStatusAndRelease(OrtStatus* ort_status, common::StatusCategory category) {
  if (ort_status == nullptr) {
    return Status::OK();
  }

  auto unique_ort_status = UniqueOrtStatus{ort_status};
  return Status(category, static_cast<common::StatusCode>(ort_status->code), &ort_status->msg[0]);
}

}  // namespace onnxruntime

#ifdef _MSC_VER
#pragma warning(pop)
#endif

ORT_API(OrtErrorCode, OrtApis::GetErrorCode, _In_ const OrtStatus* status) {
  return status->code;
}

ORT_API(const char*, OrtApis::GetErrorMessage, _In_ const OrtStatus* status) {
  return status->msg;
}

ORT_API(void, OrtApis::ReleaseStatus, _Frees_ptr_opt_ OrtStatus* value) {
  DeleteStatus(value);
}
