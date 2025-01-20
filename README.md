Install Instructions


Model:
```
curl -L https://huggingface.co/Xenova/bart-large-cnn/resolve/main/tokenizer.json -o models/bart-large-cnn/tokenizer.json
curl -L https://huggingface.co/Xenova/bart-large-cnn/resolve/main/tokenizer_config.json -o models/bart-large-cnn/tokenizer_config.json
curl -L https://huggingface.co/Xenova/bart-large-cnn/resolve/main/config.json -o models/bart-large-cnn/config.json
curl -L https://huggingface.co/Xenova/bart-large-cnn/resolve/main/onnx/encoder_model_quantized.onnx -o models/bart-large-cnn/onnx/encoder_model_quantized.onnx
curl -L https://huggingface.co/Xenova/bart-large-cnn/resolve/main/onnx/decoder_model_merged_quantized.onnx -o models/bart-large-cnn/onnx/decoder_model_merged_quantized.onnx
```

WASM Files
```
npm install @xenova/transformers 
mv node_modules/@xenova/transformers/dist/ .
```
