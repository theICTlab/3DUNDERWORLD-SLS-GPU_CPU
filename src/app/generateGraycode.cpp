// Shows the gray code pattern with option of moving the window to projector
// without chrome.
#include <GrayCode/GrayCode.hpp>
#include <iostream>
int main() {
  const auto kSize = cv::Size(1920, 1080);
  SLS::GrayCode gc(kSize.width, kSize.height);
  const std::vector<cv::Mat> images = gc.generateGrayCode();

  constexpr std::string_view win_name = "Setup Window";
  cv::namedWindow(win_name.data(), cv::WINDOW_NORMAL);

  for (size_t i = 0; i < images.size(); ++i) {
    if (i == 0) {
      std::cout << "Move window to display/projector. Any key to continue or "
                   "'q' to exit."
                << std::endl;
    } else {
      cv::setWindowProperty(win_name.data(), cv::WND_PROP_FULLSCREEN,
                            cv::WINDOW_FULLSCREEN);
    }
    cv::imshow(win_name.data(), images[i]);
    if (cv::waitKey(0) == 'q')
      break;
  }

  return EXIT_SUCCESS;
}