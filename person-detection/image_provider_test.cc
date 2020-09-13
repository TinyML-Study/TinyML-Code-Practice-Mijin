/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"

#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestImageProvider) {
  // 로깅 셋업
  tflite::MicroErrorReporter micro_error_reporter;

  // kMaxImageSize = 96 * 96 * 1
  uint8_t image_data[kMaxImageSize];

  // GetImage() 사용해 카메라 이미지 가져옴
  TfLiteStatus get_status = GetImage(&micro_error_reporter, kNumCols,
                                     kNumRows, kNumChannels, image_data);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, get_status);
  TF_LITE_MICRO_EXPECT_NE(image_data, nullptr);

  // 반환된 데이터의 메모리를 모두 읽을 수 있는지 확인
  uint32_t total = 0;
  for (int i = 0; i < kMaxImageSize; ++i) {
    total += image_data[i];
  }
}

TF_LITE_MICRO_TESTS_END
