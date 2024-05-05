extern crate image;
extern crate serde_json;
extern crate wasi_nn;

mod imagenet_classes;

use std::fs;
use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding, TensorType};

use serde_json::json;
use std::time::SystemTime;

const IMAGE_DIMENSION: u32 = 224;
const MODEL_PATH: &str = "densenet201.tflite";
const IMAGE_PATH: &str = "sample.png";

fn main() {
    println!("Starting {:?}", SystemTime::now());

    let (guess_index, guess_probability) = match run_inference() {
        Err(e) => {
            fail(&e);
            return;
        }
        Ok(t) => t,
    };

    let category = get_category(guess_index);
    let response_object = json!(
        {
            "status": "success",
            "guess_index": guess_index,
            "recognition_result": category,
            "probability": guess_probability
        }
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&response_object).unwrap()
    );
    println!("Exiting {:?}", SystemTime::now());
}

fn run_inference() -> Result<(u32, f32), &'static str> {
    let model_data = match get_model_bytes() {
        Err(e) => return Err(e),
        Ok(m) => m,
    };
    let graph = match GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU)
        .build_from_bytes(&[&model_data])
    {
        Err(_) => return Err("Failed to build model from bytes"),
        Ok(g) => g,
    };
    let mut ctx = match graph.init_execution_context() {
        Err(_) => return Err("Failed to init execution context"),
        Ok(c) => c,
    };

    let tensor_data =
        match image_to_tensor(IMAGE_PATH, IMAGE_DIMENSION, IMAGE_DIMENSION) {
            Err(_) => return Err("Failed to load image"),
            Ok(t) => t,
        };

    if let Err(_) = ctx.set_input(
        0,
        TensorType::F32,
        &[1, IMAGE_DIMENSION as usize, IMAGE_DIMENSION as usize, 3],
        &tensor_data,
    ) {
        return Err("Failed to set input to the context");
    };

    if let Err(_) = ctx.compute() {
        return Err("Failed to inference");
    };

    let mut output_buffer = vec![0f32; imagenet_classes::IMAGENET_CLASSES.len()];
    if let Err(_) = ctx.get_output(0, &mut output_buffer) {
        return Err("Failed to receive output");
    };

    let (guess_index, &guess_probability) = output_buffer
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    Ok((guess_index as u32, guess_probability))
}

fn get_model_bytes() -> Result<Vec<u8>, &'static str> { 
    let model_data = match fs::read(MODEL_PATH) {
        Err(_) => return Err("Failed to read model file"),
        Ok(m) => m,
    };

    Ok(model_data)
}

fn image_to_tensor(path: &str, height: u32, width: u32) -> Result<Vec<f32>, &'static str> {
    let img = match image::open(path) {
        Err(_) => return Err("Failed to open image file"),
        Ok(i) => i,
    };

    let resized_img = img.resize_exact(width, height, image::imageops::FilterType::Triangle);

    let rgb_img = resized_img.to_rgb8();

    // Scale pixel values according to Python's preprocess_image().
    // See https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L257.
    let input_data: Vec<f32> = rgb_img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
        .collect();

    Ok(input_data)
}

fn fail(reason: &str) {
    let response_object = json!(
        {
            "status": "error",
            "reason": reason,
        }
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&response_object).unwrap()
    );
    println!("Exiting {:?}", SystemTime::now());
}

#[inline]
fn get_category(index: u32) -> String {
    return imagenet_classes::IMAGENET_CLASSES[index as usize - 1].to_string();
}
