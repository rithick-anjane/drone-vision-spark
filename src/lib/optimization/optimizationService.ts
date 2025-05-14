import { optimizeModel } from "@/lib/optimization/modelOptimizer";
import { SimulationService, calculateFlightEndurance } from "@/lib/optimization/simulationService";
import { 
  ModelInfo, 
  OptimizationConfig, 
  OptimizationResult, 
  TargetHardwareProfile,
  HARDWARE_PROFILES
} from "@/lib/optimization/types";
import { ModelFactory } from "@/lib/optimization/modelFactory";

/**
 * Main service for handling model optimization in the UI
 */
export class OptimizationService {
  // Singleton instance
  private static instance: OptimizationService;
  
  // Model currently being optimized
  private currentModelInfo: ModelInfo | null = null;
  private currentModelBuffer: ArrayBuffer | null = null;
  
  // Callbacks for progress updates
  private progressCallbacks: ((progress: number) => void)[] = [];
  
  // Private constructor (singleton pattern)
  private constructor() {}
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): OptimizationService {
    if (!OptimizationService.instance) {
      OptimizationService.instance = new OptimizationService();
    }
    return OptimizationService.instance;
  }
  
  /**
   * Register a callback for progress updates
   */
  public onProgress(callback: (progress: number) => void): () => void {
    this.progressCallbacks.push(callback);
    
    // Return a function to unregister the callback
    return () => {
      this.progressCallbacks = this.progressCallbacks.filter(cb => cb !== callback);
    };
  }
  
  /**
   * Update progress and notify all listeners
   */
  private updateProgress(progress: number): void {
    this.progressCallbacks.forEach(callback => callback(progress));
  }
  
  /**
   * Load a model file
   */
  public async loadModelFile(file: File): Promise<ModelInfo> {
    // Read the file as ArrayBuffer
    const buffer = await this.readFileAsArrayBuffer(file);
    
    // Create a model info object
    const format = file.name.split('.').pop() as string;
    const modelInfo: ModelInfo = {
      id: this.generateId(),
      name: file.name,
      format,
      architecture: this.detectArchitecture(file.name),
      timestamp: Date.now()
    };
    
    // Store the current model
    this.currentModelBuffer = buffer;
    this.currentModelInfo = modelInfo;
    
    return modelInfo;
  }
  
  /**
   * Load a pretrained model by ID
   */
  public async loadPretrainedModel(modelId: string): Promise<ModelInfo> {
    // Get the model info
    const pretrainedModel = ModelFactory.getModelById(modelId);
    if (!pretrainedModel) {
      throw new Error(`Pretrained model with ID "${modelId}" not found`);
    }
    
    // Update progress to indicate loading
    this.updateProgress(10);
    
    // Simulate loading the model
    const buffer = await ModelFactory.simulateModelLoading(modelId);
    
    // Update progress to indicate done loading
    this.updateProgress(20);
    
    // Create a model info object
    const modelInfo = ModelFactory.createModelInfo(pretrainedModel);
    
    // Store the current model
    this.currentModelBuffer = buffer;
    this.currentModelInfo = modelInfo;
    
    return modelInfo;
  }
  
  /**
   * Optimize the current model with the given configuration
   */
  public async optimizeCurrentModel(config: OptimizationConfig): Promise<OptimizationResult> {
    if (!this.currentModelInfo || !this.currentModelBuffer) {
      throw new Error("No model loaded. Please load a model first.");
    }
    
    // Update progress to indicate starting optimization
    this.updateProgress(25);
    
    // Create a worker to simulate the optimization process
    const result = await this.simulateOptimizationProcess(
      this.currentModelBuffer,
      this.currentModelInfo,
      config
    );
    
    return result;
  }
  
  /**
   * Simulate the optimization process with progress updates
   */
  private async simulateOptimizationProcess(
    modelBuffer: ArrayBuffer,
    modelInfo: ModelInfo,
    config: OptimizationConfig
  ): Promise<OptimizationResult> {
    // This simulates a real optimization process with progress updates
    // In a real implementation, this would use web workers or backend API
    
    // Simulate the optimization steps
    const steps = [
      { name: "Analyzing model", progress: 30, delay: 1000 },
      { name: "Preparing optimization", progress: 40, delay: 800 },
    ];
    
    if (config.pruningEnabled) {
      steps.push({ name: "Applying pruning", progress: 60, delay: 1500 });
    }
    
    if (config.quantizationEnabled) {
      steps.push({ name: "Applying quantization", progress: 80, delay: 1200 });
    }
    
    if (config.fineTuningEnabled) {
      steps.push({ name: "Fine-tuning model", progress: 90, delay: 2000 });
    }
    
    steps.push({ name: "Evaluating performance", progress: 95, delay: 500 });
    steps.push({ name: "Finalizing optimization", progress: 100, delay: 300 });
    
    // Execute each step with the specified delay
    for (const step of steps) {
      this.updateProgress(step.progress);
      await new Promise(resolve => setTimeout(resolve, step.delay));
    }
    
    // Use the actual optimization function (which is currently a simulation in our case)
    return await optimizeModel(modelBuffer, modelInfo, config);
  }
  
  /**
   * Simulate model performance on a specific hardware profile
   */
  public simulateModelPerformance(
    result: OptimizationResult,
    hardwareProfileId: string,
    droneSpecs?: {
      batteryCapacity: number; // mAh
      weight: number; // grams
      efficiency: number; // g/W
    }
  ): {
    performance: {
      original: any;
      optimized: any;
    };
    flightEndurance?: {
      original: any;
      optimized: any;
    };
  } {
    // Get the hardware profile
    const hardwareProfile = HARDWARE_PROFILES[hardwareProfileId] || HARDWARE_PROFILES["generic-drone"];
    
    // Simulate performance on the target hardware
    const originalPerformance = SimulationService.simulatePerformance(
      result.originalModel.size,
      result.originalModel.inferenceTime,
      hardwareProfile
    );
    
    const optimizedPerformance = SimulationService.simulatePerformance(
      result.optimizedModel.size,
      result.optimizedModel.inferenceTime,
      hardwareProfile
    );
    
    // If drone specs are provided, calculate flight endurance
    let flightEndurance;
    if (droneSpecs) {
      const originalEndurance = calculateFlightEndurance(
        originalPerformance,
        hardwareProfile,
        droneSpecs
      );
      
      const optimizedEndurance = calculateFlightEndurance(
        optimizedPerformance,
        hardwareProfile,
        droneSpecs
      );
      
      flightEndurance = {
        original: originalEndurance,
        optimized: optimizedEndurance
      };
    }
    
    return {
      performance: {
        original: originalPerformance,
        optimized: optimizedPerformance
      },
      flightEndurance
    };
  }
  
  /**
   * Generate a unique ID for models
   */
  private generateId(): string {
    return 'model_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
  }
  
  /**
   * Try to detect model architecture from filename
   */
  private detectArchitecture(filename: string): string {
    filename = filename.toLowerCase();
    
    if (filename.includes('yolo')) {
      return 'yolo';
    } else if (filename.includes('mobilenet') || filename.includes('ssd')) {
      return 'mobilenet';
    } else if (filename.includes('efficient')) {
      return 'efficientdet';
    } else {
      return 'generic';
    }
  }
  
  /**
   * Read a file as ArrayBuffer
   */
  private readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  }
  
  /**
   * Get example optimization configurations
   */
  public getExampleConfigurations(): Record<string, OptimizationConfig> {
    return {
      "balanced": {
        pruningEnabled: true,
        pruningType: "structured",
        compressionLevel: 50,
        quantizationEnabled: true,
        quantizationType: "post-training",
        quantizationPrecision: 8,
        fineTuningEnabled: true,
        fineTuningSteps: 100,
        targetHardware: HARDWARE_PROFILES["generic-drone"]
      },
      "max-speed": {
        pruningEnabled: true,
        pruningType: "structured",
        compressionLevel: 70,
        quantizationEnabled: true,
        quantizationType: "post-training",
        quantizationPrecision: 8,
        fineTuningEnabled: false,
        fineTuningSteps: 0,
        targetHardware: HARDWARE_PROFILES["generic-drone"]
      },
      "max-accuracy": {
        pruningEnabled: true,
        pruningType: "unstructured",
        compressionLevel: 30,
        quantizationEnabled: true,
        quantizationType: "quantization-aware",
        quantizationPrecision: 16,
        fineTuningEnabled: true,
        fineTuningSteps: 200,
        targetHardware: HARDWARE_PROFILES["generic-drone"]
      },
      "tpu-optimized": {
        pruningEnabled: true,
        pruningType: "structured",
        compressionLevel: 60,
        quantizationEnabled: true,
        quantizationType: "post-training",
        quantizationPrecision: 8,
        fineTuningEnabled: true,
        fineTuningSteps: 50,
        targetHardware: HARDWARE_PROFILES["coral-tpu"]
      }
    };
  }
} 