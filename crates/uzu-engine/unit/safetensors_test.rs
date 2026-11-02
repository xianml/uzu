use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{Backend, Context},
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::weight_matrix::WeightMatrix,
    safetensors::{ParameterLoader, ParameterLoaderError},
    tests::helpers::for_each_backend,
};

#[uzu_test]
fn trellis_async_loading_preserves_tensor_checks_and_zeroes_tail() {
    let header = serde_json::json!({
        "codes": {"dtype": "U8", "shape": [2, 32], "data_offsets": [0, 64]},
        "scales": {"dtype": "F32", "shape": [2], "data_offsets": [64, 72]},
        "codebook": {"dtype": "F32", "shape": [5], "data_offsets": [72, 92]}
    });
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    let codes: Vec<u8> = (1..=64).collect();
    file.extend_from_slice(&codes);
    file.extend_from_slice(bytemuck::cast_slice(&[1.0f32; 7]));
    let path = std::env::temp_dir().join(format!("uzu-pr899-loading-{}.safetensors", std::process::id()));
    std::fs::write(&path, file).unwrap();

    for_each_backend!(|B| {
        block_on(async {
            let context = <B as Backend>::Context::new().await.unwrap();
            let loader = ParameterLoader::<B>::new(&path, &context).await.unwrap();
            let tree = loader.tree();
            // A padded allocation must still validate the tensor's unpadded shape and byte count.
            assert!(matches!(
                tree.read_buffer_with_tail("codes", &[2, 33], DataType::U8, 16).await,
                Err(ParameterLoaderError::InvalidTensor { .. })
            ));
            assert!(loader.assert_all_tensors_read().is_err());
            let matrix = WeightMatrix::load(
                2,
                64,
                DataType::BF16,
                serde_json::from_value::<AnyWeightMatrixSpec>(serde_json::json!({
                    "type": "QtipGaussianSpec",
                    "vector_width": 4,
                    "transition_bits": 8,
                    "restart_columns": 64
                }))
                .unwrap(),
                Layout::OutputInput,
                &tree,
            )
            .await
            .unwrap();
            let loaded = context.copyout::<u8>(matrix.values()).await.unwrap();
            assert_eq!(&loaded[..64], &codes);
            assert_eq!(&loaded[64..], &[0; 16]);
            loader.assert_all_tensors_read().unwrap();
        });
    });
    std::fs::remove_file(path).unwrap();
}

#[uzu_test]
fn trellis_model_loading_future_is_send() {
    fn check<B: Backend>(engine: &std::sync::Arc<crate::engine::Engine<B>>) {
        fn assert_send<T: Send>(_: T) {}
        assert_send(engine.load_language_model(std::path::Path::new("unused")));
    }
    // Check the generic backend contract without starting model IO.
    let _ = check::<crate::backends::cpu::Cpu>;
}
