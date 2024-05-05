extern crate image;
extern crate serde_json;
extern crate tflite;

mod imagenet_classes;

use std::cmp::Ordering;
use std::result::Result;

use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};

use serde_json::json;
use std::time::SystemTime;

const IMAGE_DIMENSION: u32 = 224;

fn main() {
    println!("Starting {:?}", SystemTime::now());

    let binding = FlatBufferModel::build_from_file("densenet201.tflite");
    let model = match &binding {
        Err(_) => {
            fail("Failed to load model");
            return;
        }
        Ok(m) => m,
    };
    let (guess_index, guess_probability) = match run_inference(model) {
        Err(e) => {
            fail(&e);
            return;
        }
        Ok(t) => t,
    };

    let category = get_category(guess_index);
    let response_object = json!(
        {
            "status": "success", "guess_index": guess_index,
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

fn run_inference(model: &FlatBufferModel) -> Result<(u32, f32), &'static str> {
    let resolver = BuiltinOpResolver::default();

    let builder = match InterpreterBuilder::new(model, &resolver) {
        Err(_) => return Err("Failed to build new interpreter model"),
        Ok(b) => b,
    };

    let mut interpreter = match builder.build() {
        Err(_) => return Err("Failed to build new model"),
        Ok(i) => i,
    };

    if let Err(_) = interpreter.allocate_tensors() {
        return Err("Failed to allocate tensors");
    }

    let inputs = interpreter.inputs().to_vec();
    //assert_eq!(inputs.len(), 1);

    let input_index = inputs[0];

    let outputs = interpreter.outputs().to_vec();
    //assert_eq!(outputs.len(), 1);

    let output_index = outputs[0];

    let img = match image::open("sample.png") {
        Err(_) => return Err("Failed to open image"),
        Ok(i) => i,
    };

    let resized_img = img.resize(
        IMAGE_DIMENSION,
        IMAGE_DIMENSION,
        image::imageops::FilterType::Triangle,
    );
    let rgb_img = resized_img.to_rgb8();

    // Scale pixel values according to Python's preprocess_image().
    // See https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L257.
    let input_data: Vec<f32> = rgb_img
        .pixels()
        .flat_map(|p| {
            p.0.iter().map(|&v| v as f32 / 127.5 - 1.0)
        })
        .collect();
    //assert_eq!(
    //    input_data.len(),
    //    (1 * IMAGE_DIMENSION * IMAGE_DIMENSION * 3) as usize,
    //    "Mismatch in the number of elements for the input tensor."
    //);

    let vec = match interpreter.tensor_data_mut::<f32>(input_index) {
        Err(_) => return Err("Failed tensor_data_mut"),
        Ok(v) => v,
    };

    vec.copy_from_slice(&input_data);

    if let Err(_) = interpreter.invoke() {
        return Err("Failed to inference the model");
    }

    let output: &[f32] = match interpreter.tensor_data::<f32>(output_index) {
        Err(_) => return Err("Failed tensor_data"),
        Ok(o) => o,
    };

    let (guess_index, &guess_probability) = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap();

    Ok((guess_index as u32, guess_probability))
}

#[inline]
fn get_category(index: u32) -> String {
    return imagenet_classes::IMAGENET_CLASSES[index as usize - 1].to_string();
}

