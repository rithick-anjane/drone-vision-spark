/**
 * Information about the model being optimized
 */
export interface ModelInfo {
  id: string;
  name: string;
  format: "pt" | "onnx" | "tflite" | string;
  architecture: string;
  version?: string;
  purpose?: string;
  classes?: string[];
  timestamp: number;
}

/**
 * Pruning method types
 */
export type PruningType = "unstructured" | "structured" | "group-channel";

/**
 * Quantization method types
 */
export type QuantizationType = "post-training" | "quantization-aware" | "optimal-brain";

/**
 * Advanced optimization options
 */
export interface AdvancedOptions {
  // Pruning advanced options
  gradualPruning?: boolean;
  pruningSchedule?: "linear" | "exponential" | "cubic";
  
  // Quantization advanced options
  useCalibrationData?: boolean;
  perChannelQuantization?: boolean;
  
  // Fine-tuning advanced options
  learningRate?: number;
  useKnowledgeDistillation?: boolean;
  
  // Post-processing options
  applyLayerFusion?: boolean;
  optimizeMemoryLayout?: boolean;
}

/**
 * Configuration for optimization process
 */
export interface OptimizationConfig {
  // Pruning options
  pruningEnabled: boolean;
  pruningType: PruningType;
  compressionLevel: number; // 0-100, percentage of weights to prune
  
  // Quantization options
  quantizationEnabled: boolean;
  quantizationType: QuantizationType;
  quantizationPrecision: 8 | 16 | 32; // Bit-width for quantization
  
  // Fine-tuning options
  fineTuningEnabled: boolean;
  fineTuningSteps: number;
  
  // Target hardware profile
  targetHardware: TargetHardwareProfile;
  
  // Export options
  exportFormat?: "pt" | "onnx" | "tflite" | string;
  
  // Advanced optimization options
  advancedOptions?: AdvancedOptions;
}

/**
 * Hardware constraints for target devices
 */
export interface TargetHardwareProfile {
  name: string;
  cpu: {
    cores: number;
    frequency: number; // MHz
  };
  memory: number; // MB
  powerConsumption: number; // Watts
  accelerator?: {
    type: "GPU" | "TPU" | "NPU" | "None";
    memory?: number; // MB
  };
}

/**
 * Performance metrics for a model
 */
export interface ModelMetrics {
  size: number; // Model size in MB
  accuracy: number; // Accuracy percentage (0-100)
  inferenceTime: number; // Inference time in milliseconds
  memoryUsage?: number; // Memory usage in MB
  powerConsumption?: number; // Power usage in Watts
}

/**
 * Result of optimization process
 */
export interface OptimizationResult {
  originalModel: ModelMetrics;
  optimizedModel: ModelMetrics;
  modelBuffer: ArrayBuffer; // The optimized model binary data
  reductionRatio: {
    size: number; // How many times smaller (e.g., 2 = half size)
    inferenceTime: number; // How many times faster
  };
  optimizationSteps: string[]; // List of steps applied
  targetHardware: string; // Name of the target hardware
}

/**
 * Predefined hardware profiles for common drone platforms
 */
export const HARDWARE_PROFILES: Record<string, TargetHardwareProfile> = {
  "generic-drone": {
    name: "Generic Drone",
    cpu: { cores: 4, frequency: 1200 },
    memory: 2048,
    powerConsumption: 2.5
  },
  "raspberry-pi": {
    name: "Raspberry Pi 4",
    cpu: { cores: 4, frequency: 1500 },
    memory: 4096,
    powerConsumption: 3.0
  },
  "jetson-nano": {
    name: "NVIDIA Jetson Nano",
    cpu: { cores: 4, frequency: 1400 },
    memory: 4096,
    powerConsumption: 5.0,
    accelerator: {
      type: "GPU",
      memory: 2048
    }
  },
  "intel-ncs": {
    name: "Intel Neural Compute Stick 2",
    cpu: { cores: 1, frequency: 700 },
    memory: 512,
    powerConsumption: 1.0,
    accelerator: {
      type: "NPU",
      memory: 4096
    }
  },
  "coral-tpu": {
    name: "Google Coral TPU",
    cpu: { cores: 1, frequency: 500 },
    memory: 256,
    powerConsumption: 0.5,
    accelerator: {
      type: "TPU",
      memory: 1024
    }
  }
}; 