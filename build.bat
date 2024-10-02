trtexec.exe --onnx=%1 --saveEngine=model.engine --minShapes=input:16x256x4x4 --optShapes=input:16x256x4x4 --maxShapes=input:16x256x4x4 --fp16
