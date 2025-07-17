# B2-EdgeAI
programming project of group B2 in EdgeAI

There are detailed README.md file in each of the main folder. see it for more information

In this project , we worked on the federated learning code, that can be used to implement and train the object detection model using the  decentralized federated prinicple.
We also fine tuned a pretrained Yolo11n model, to recognise following classes more precisely :  "bus", "coupe", "crossover", "hatchback", "jeep", "mpv",
  "pickup-truck", "sedan", "suv", "taxi", "truck", "van", "vehicle", "wagon".


### Benchmark

| Format              | Device                            | Inference Speed (ms) |
|---------------------|------------------------------------|-----------------------|
| PyTorch             | NVIDIA GeForce RTX 3060 Ti         | 12.13                 |
| PyTorch             | NVIDIA 940MX                       | 225.6                 |
| ONNX                | (CPU) Intel i5-7200U               | 35.08                 |
| TensorRT `.engine` (FP32) | Jetson Nano (JetPack 4)        | 352.3                  |

---

### Dataset
In this project, our challenge was to utilize the **AI City Challenge 2023 Track 2** dataset.





