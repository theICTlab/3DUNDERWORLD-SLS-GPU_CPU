// Utility to project gray code patterns, synchronize cameras and saving frames.
#include <filesystem>
#include "GrayCode/GrayCode.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "opencv2/opencv.hpp"

ABSL_FLAG(int32_t, left_cam, 0, "Left camera ID");
ABSL_FLAG(int32_t, right_cam, 2, "Right camera ID");
ABSL_FLAG(std::string, left_cam_dir, "/tmp/left_cam/dataset1",
          "Directory of the left cam");
ABSL_FLAG(std::string, right_cam_dir, "/tmp/right_cam/dataset1",
          "Directory of the right cam");

absl::Status ConfigureCamera(cv::VideoCapture& cap) {
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  cap.set(cv::CAP_PROP_EXPOSURE, -6);
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
}

absl::StatusOr<cv::Mat> GetFrame(cv::VideoCapture& cap,
                                 size_t flush_count = 5) {
  cv::Mat frame;
  for (size_t i = 0; i < flush_count; ++i) {
    if (!cap.grab()) {
      return absl::InternalError("Failed to grab frames.");
    }
  }
  if (!cap.retrieve(frame)) {
    return absl::InternalError("Failed to retrieve frame.");
  }
  return frame;
}

absl::Status SaveFrame(const cv::Mat& frame, const std::filesystem::path& dir,
                       std::string_view filename) {
  std::filesystem::path file_path = dir / filename;
  if (!cv::imwrite(file_path.string(), frame)) {
    return absl::InternalError(
        absl::StrFormat("Failed to save frame to '%s'.", file_path.string()));
  }
  return absl::OkStatus();
}

absl::Status CapturePair(cv::VideoCapture& left_cap,
                         cv::VideoCapture& right_cap,
                         const std::filesystem::path& left_dir,
                         const std::filesystem::path& right_dir, size_t i,
                         size_t flush_count = 10) {
  cv::Mat frame;
  absl::StatusOr<cv::Mat> frame_or;
  if (frame_or = GetFrame(left_cap, flush_count); !frame_or.ok()) {
    return frame_or.status();
  }
  frame = frame_or.value();
  if (auto status = SaveFrame(frame, left_dir, absl::StrFormat("%04d.jpg", i));
      !status.ok()) {
    return status;
  }
  if (frame_or = GetFrame(left_cap, flush_count); !frame_or.ok()) {
    return frame_or.status();
  }
  frame = frame_or.value();
  if (auto status = SaveFrame(frame, right_dir, absl::StrFormat("%04d.jpg", i));
      !status.ok()) {
    return status;
  }
  if (frame_or = GetFrame(left_cap); !frame_or.ok()) {
    return frame_or.status();
  }
  return absl::OkStatus();
}

absl::Status Run() {
  // Create output directories if they do not exist.
  if (!std::filesystem::exists(absl::GetFlag(FLAGS_left_cam_dir))) {
    if (!std::filesystem::create_directories(absl::GetFlag(FLAGS_left_cam_dir)))
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to create '%s directory.",
                          absl::GetFlag(FLAGS_left_cam_dir)));
  }
  std::filesystem::path left_dir(absl::GetFlag(FLAGS_left_cam_dir));
  if (!std::filesystem::exists(absl::GetFlag(FLAGS_right_cam_dir))) {
    if (!std::filesystem::create_directories(
            absl::GetFlag(FLAGS_right_cam_dir)))
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to create '%s' directory.",
                          absl::GetFlag(FLAGS_right_cam_dir)));
  }
  std::filesystem::path right_dir(absl::GetFlag(FLAGS_right_cam_dir));
  // Set up gray code
  const auto kSize = cv::Size(1920, 1080);
  SLS::GrayCode gc(kSize.width, kSize.height);
  const std::vector<cv::Mat> images = gc.generateGrayCode();
  CHECK(images.size() > 0);
  LOG(INFO) << absl::StreamFormat(
      "Projector size: %ix%i, number of patterns: %i", kSize.width,
      kSize.height, images.size());
  // Set up projector window.
  constexpr std::string_view win_name = "Setup Window";
  cv::namedWindow(win_name.data(), cv::WINDOW_NORMAL);
  LOG(INFO) << "Move window to display/projector. Press any key to "
               "continue or 'q' to exit.";
  cv::imshow(win_name.data(), images[0]);
  if (cv::waitKey(0) == 'q') {
    return absl::OkStatus();
  }
  // Start capturing.
  cv::VideoCapture left_cap;
  cv::VideoCapture right_cap;
  for (size_t i = 0; i < images.size(); ++i) {
    cv::setWindowProperty(win_name.data(), cv::WND_PROP_FULLSCREEN,
                          cv::WINDOW_FULLSCREEN);
    cv::imshow(win_name.data(), images[i]);
    if (i == 0) {
      // Starting cameras
      left_cap.open(absl::GetFlag(FLAGS_left_cam));
      if (!left_cap.isOpened()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Failed to open left camera: %i", absl::GetFlag(FLAGS_left_cam)));
      }
      right_cap.open(absl::GetFlag(FLAGS_right_cam));
      if (!right_cap.isOpened()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Failed to open right camera: %i", absl::GetFlag(FLAGS_right_cam)));
      }
     }
    if (cv::waitKey(/*delay=*/0) == 'q') break;
    if (auto status = CapturePair(left_cap, right_cap, left_dir, right_dir, i);
        !status.ok()) {
      return status;
    }
  }
  LOG(INFO) << absl::StreamFormat("Saved %d frames into %s and %s",
                                  images.size(), left_dir.string(),
                                  right_dir.string());
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (const auto status = Run(); !status.ok()) {
    LOG(ERROR) << status.message();
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Done";
  return EXIT_SUCCESS;
}