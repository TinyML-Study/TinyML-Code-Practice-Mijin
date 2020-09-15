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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_optional_debug_tools.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

constexpr int tensor_arena_size = 90 * 1024;
uint8_t tensor_arena[tensor_arena_size];

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Typical setup
  // 로깅 셋업
  tflite::MicroErrorReporter micro_error_reporter;

  // 모델 맵핑
  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // 필요한 operation 등록
  tflite::MicroMutableOpResolver<3> micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddAveragePool2D();
  micro_mutable_op_resolver.AddConv2D();
  micro_mutable_op_resolver.AddDepthwiseConv2D();

  // 모델을 run 할 interpreter build
  tflite::MicroInterpreter interpreter(model, micro_mutable_op_resolver,
                                       tensor_arena, tensor_arena_size,
                                       &micro_error_reporter);
  interpreter.AllocateTensors();

  // Inspect the input tensor
  // 모델의 input으로 사용할 메모리 영역에 대한 정보 가져오기
  TfLiteTensor* input = interpreter.input(0);

  // Input이 적절한 property를 갖고 있는지 확인
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  // 테스트를 위해, 사람 이미지를 input tensor로 copy
  const uint8_t* person_data = g_person_data;
  for (size_t i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = person_data[i];
  }

  // 위 input을 사용해 모델 run 한 후, 성공하는 지 확인
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Output tensor가 적절한 property를 갖고 있는지 확인
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  // kCategoryCount = 3 (index 0: unused, index 1: person, index 2: no person)
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // Person score가 더 높은지 확인
  uint8_t person_score = output->data.uint8[kPersonIndex];
  uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  TF_LITE_REPORT_ERROR(&micro_error_reporter,
                       "person data. person score: %d, no person score: %d\n",
                       person_score, no_person_score);
  TF_LITE_MICRO_EXPECT_GT(person_score, no_person_score);

  // 사람이 아닌 이미지를 input으로 테스트
  const uint8_t* no_person_data = g_no_person_data;
  for (size_t i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = no_person_data[i];
  }

  // 위 input을 사용해 모델 run
  invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Output tensor가 적절한 property를 갖고 있는지 확인
  output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // No person score가 더 높은지 확인
  person_score = output->data.uint8[kPersonIndex];
  no_person_score = output->data.uint8[kNotAPersonIndex];
  TF_LITE_REPORT_ERROR(&micro_error_reporter,
      "no person data. no person score: %d, person score: %d\n",
      no_person_score, person_score);
  TF_LITE_MICRO_EXPECT_GT(no_person_score, person_score);
}

TF_LITE_MICRO_TESTS_END