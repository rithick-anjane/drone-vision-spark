import { ModelInfo } from "@/lib/optimization/types";

/**
 * Factory for creating and working with pretrained models
 */
export class ModelFactory {
  /**
   * List of available pretrained models
   */
  static readonly PRETRAINED_MODELS: PretrainedModel[] = [
    {
      id: "yolov5s",
      name: "YOLOv5s",
      architecture: "yolo",
      version: "5.0",
      description: "Lightweight version of YOLOv5 optimized for speed.",
      defaultFormat: "pt",
      size: 14.1, // MB
      accuracy: 36.7, // mAP@0.5:0.95
      inferenceTime: 6.4, // ms on GPU
      params: 7.2, // Million parameters
      url: "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
    },
    {
      id: "yolov8n",
      name: "YOLOv8n",
      architecture: "yolo",
      version: "8.0",
      description: "Nano version of YOLOv8, extremely fast for mobile deployment.",
      defaultFormat: "pt",
      size: 6.3, // MB
      accuracy: 37.3, // mAP@0.5:0.95
      inferenceTime: 3.2, // ms on GPU
      params: 3.2, // Million parameters
      url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    },
    {
      id: "mobilenet-ssdv2",
      name: "MobileNet SSDv2",
      architecture: "mobilenet",
      version: "2.0",
      description: "MobileNetV2 with SSD, designed for mobile and embedded vision applications.",
      defaultFormat: "tflite",
      size: 8.9, // MB
      accuracy: 22.1, // mAP@0.5:0.95
      inferenceTime: 5.8, // ms on mobile device
      params: 4.3, // Million parameters
      url: "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    },
    {
      id: "efficientdet-d0",
      name: "EfficientDet-D0",
      architecture: "efficientdet",
      version: "1.0",
      description: "Smallest variant of EfficientDet, excellent accuracy-to-size ratio.",
      defaultFormat: "tflite",
      size: 15.6, // MB
      accuracy: 33.8, // mAP@0.5:0.95
      inferenceTime: 9.2, // ms
      params: 3.9, // Million parameters
      url: "https://github.com/google/automl/tree/master/efficientdet"
    },
    {
      id: "yolor-csp",
      name: "YOLOR-CSP",
      architecture: "yolor",
      version: "1.0",
      description: "Unified network for object detection, optimized for accuracy.",
      defaultFormat: "pt",
      size: 32.4, // MB
      accuracy: 52.8, // mAP@0.5:0.95
      inferenceTime: 17.8, // ms
      params: 52.2, // Million parameters
      url: "https://github.com/WongKinYiu/yolor/releases"
    }
  ];
  
  /**
   * Get a model by its ID
   */
  static getModelById(id: string): PretrainedModel | undefined {
    return this.PRETRAINED_MODELS.find(model => model.id === id);
  }
  
  /**
   * Get all models of a specific architecture
   */
  static getModelsByArchitecture(architecture: string): PretrainedModel[] {
    return this.PRETRAINED_MODELS.filter(model => model.architecture === architecture);
  }
  
  /**
   * Convert a pretrained model to ModelInfo
   */
  static createModelInfo(model: PretrainedModel): ModelInfo {
    return {
      id: model.id,
      name: model.name,
      format: model.defaultFormat,
      architecture: model.architecture,
      version: model.version,
      purpose: "object-detection",
      classes: ["person", "car", "truck", "bicycle", "motorcycle", "bus", "boat", "traffic light", "fire hydrant", "stop sign"],
      timestamp: Date.now()
    };
  }
  
  /**
   * Simulate loading a model (for demo purposes)
   */
  static async simulateModelLoading(modelId: string): Promise<ArrayBuffer> {
    // In a real implementation, this would download and load the actual model
    // For now, we just simulate a delay and return a dummy buffer
    
    await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate loading time
    
    // Create a dummy binary buffer
    return new ArrayBuffer(1024 * 1024 * 10); // 10MB dummy buffer
  }
}

/**
 * Pretrained model information
 */
export interface PretrainedModel {
  id: string;
  name: string;
  architecture: string;
  version: string;
  description: string;
  defaultFormat: "pt" | "onnx" | "tflite";
  size: number; // MB
  accuracy: number; // mAP
  inferenceTime: number; // ms
  params: number; // Million parameters
  url: string;
} 