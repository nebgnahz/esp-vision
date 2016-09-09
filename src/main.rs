//! Example applications that demonstrate how to use OpenCV and ESP.
//!
//! - OpenCV is wrapped by a Rust library (currently a private repository)
//! - [ESP](https://github.com/damellis/ESP): Example-based Sensor Predictions
//!
//! This application runs OpenCV
//! [CAMShift](http://docs.opencv.org/3.1.0/db/df8/tutorial_py_meanshift.html)
//! algorithm. Users select a region to track and the centroid of the tracked
//! region is sent over to ESP (through a TCP connection).
//!
//! # Getting Started
//!
//! Below is a possible ESP example that you can use (note the TCP port `8001`).
//!
//! ```c++
//! #include <ESP.h>
//!
//! TcpInputStream stream(8001, 2);
//! GestureRecognitionPipeline pipeline;
//!
//! int timeout = 500;      // milliseconds
//! double null_rej = 0.4;
//!
//! void setup() {
//!     stream.setLabelsForAllDimensions({"x", "y", "z"});
//!     useInputStream(stream);
//!
//!     DTW dtw(false, true, null_rej);
//!     dtw.enableTrimTrainingData(true, 0.1, 75);
//!
//!     pipeline.setClassifier(dtw);
//!     pipeline.addPostProcessingModule(ClassLabelTimeoutFilter(timeout));
//!     usePipeline(pipeline);
//!
//!     registerTuneable(
//!         null_rej, 0.1, 5.0, "Variability",
//!         "How different from the training data a new gesture can be and "
//!         "still be considered the same gesture. The higher the number, the "
//!         "more different it can be.",
//!         [](double new_null_rej) {
//!             pipeline.getClassifier()->setNullRejectionCoeff(new_null_rej);
//!             pipeline.getClassifier()->recomputeNullRejectionThresholds();
//!         });
//!
//!     registerTuneable(
//!         timeout, 1, 3000, "Timeout",
//!         "How long (in milliseconds) to wait after recognizing a "
//!         "gesture before recognizing another one.",
//!         [](double new_timeout) {
//!             ClassLabelTimeoutFilter* filter =
//!                 dynamic_cast<ClassLabelTimeoutFilter*>(
//!                     pipeline.getPostProcessingModule(0));
//!             assert(filter != nullptr);
//!             filter->setTimeoutDuration(new_timeout);
//!         });
//! }
//!```
//!
//! After running the ESP example, in this application, type `cargo run` would
//! bring up the application.
//!
//! Enjoy watching yourself :)
extern crate rust_vision;
use rust_vision::*;
use std::io::prelude::*;
use std::net::TcpStream;

/// `SelectionStatus` tracks the region that users have selected for tracking.
struct SelectionStatus {
    selection: Rect,
    status: bool,
}

/// Mouse callback function. This gets called whenever a mouse event
/// happens. Specifically in the implementation here we are populating the
/// `SelectionStatus` struct so that CAMShift will track the right region.
fn on_mouse(e: i32, x: i32, y: i32, _: i32, data: MouseCallbackData) {
    let event: MouseEventTypes = unsafe { std::mem::transmute(e as u8) };
    match event {
        MouseEventTypes::LButtonDown => {
            let ss = data as *mut SelectionStatus;
            let mut selection = unsafe { &mut (*ss).selection };
            selection.x = x;
            selection.y = y;
        }
        MouseEventTypes::LButtonUp => {
            let ss = data as *mut SelectionStatus;
            let mut selection = unsafe { &mut (*ss).selection };
            let mut status = unsafe { &mut (*ss).status };
            selection.width = x - selection.x;
            selection.height = y - selection.y;

            if selection.width > 0 && selection.height > 0 {
                *status = true;
            }
        }
        _ => {}
    }
}

/// The entry point to the application. Click into
/// [source](../src/esp_vision/src/main.rs.html#103-180) for more information.
fn main() {
    let mut stream = TcpStream::connect("127.0.0.1:8001")
        .ok()
        .expect("The server is not on");

    let mut selection_status = SelectionStatus {
        selection: Rect::default(),
        status: false,
    };
    let ss_ptr = &mut selection_status as *mut SelectionStatus;

    let cap = VideoCapture::new(0);
    assert!(cap.is_open());

    highgui_named_window("Window", WindowFlags::WindowAutosize);
    highgui_set_mouse_callback("Window", on_mouse, ss_ptr as MouseCallbackData);

    let mut m = Mat::new();
    let mut is_tracking = false;

    let mut hist = Mat::new();
    let hsize = 16;
    let hranges = [0_f32, 180_f32];
    let phranges: [*const f32; 1] = [&hranges[0] as *const f32];
    let mut track_window = Rect::default();

    loop {
        cap.read(&m);
        m.flip(FlipCode::YAxis);

        let hsv = m.cvt_color(ColorConversionCodes::BGR2HSV);

        let ch = [0, 0];
        let hue = hsv.mix_channels(1, 1, &ch[0] as *const i32, 1);
        let mask =
            hsv.in_range(Scalar::new(0, 30, 10, 0),
                         Scalar::new(180, 256, 256, 0));

        if selection_status.status {
            println!("Initialize tracking, setting up CAMShift search");
            let selection = selection_status.selection;
            let roi = hue.roi(selection);
            let maskroi = mask.roi(selection);

            let raw_hist = roi.calc_hist(std::ptr::null(),
                                         maskroi,
                                         1,
                                         &hsize,
                                         &phranges[0] as *const *const f32);
            hist =
                raw_hist.normalize(0 as f64, 255 as f64, NormTypes::NormMinMax);

            track_window = selection;
            m.rectangle(selection);
            selection_status.status = false;
            is_tracking = true;
        }

        if is_tracking {
            let mut back_project = hue.calc_back_project(std::ptr::null(),
                                   &hist,
                                   &phranges[0] as *const *const f32);
            back_project.logic_and(mask);
            let criteria = TermCriteria::new(TermType::Count, 10, 1 as f64);
            let track_box = back_project.camshift(track_window, &criteria);

            let bounding = track_box.bounding_rect();
            m.rectangle(bounding);
            let msg: String = (bounding.x + bounding.width / 2).to_string() +
                              " " +
                              &(bounding.y + bounding.height / 2).to_string() +
                              " \n";
            stream.write(msg.as_bytes()).ok();
        }

        m.show("Window", 30);
    }
}
