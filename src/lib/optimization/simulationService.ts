import { ModelMetrics, TargetHardwareProfile } from "@/lib/optimization/types";

/**
 * Simulates how the optimized model would perform on real hardware
 */
export class SimulationService {
  /**
   * Simulates model performance under the given hardware constraints
   */
  static simulatePerformance(
    modelSize: number, // MB
    baseInferenceTime: number, // ms
    targetHardware: TargetHardwareProfile
  ): ModelMetrics {
    // Calculate adjustments based on hardware constraints
    
    // Memory Usage: Typically model size + runtime overhead
    const memoryOverhead = 1.5; // 50% overhead for runtime
    const memoryUsage = modelSize * memoryOverhead;
    
    // Is the model too large for the device?
    const memoryConstraint = Math.min(1, targetHardware.memory / memoryUsage);
    
    // Calculate inference time with hardware constraints
    // Using a simplified model considering CPU frequency and cores
    const cpuPower = targetHardware.cpu.frequency * Math.sqrt(targetHardware.cpu.cores);
    const cpuFactor = 1500 / cpuPower; // Normalized to a reference CPU
    
    // Accelerator speedup factor (if present)
    let acceleratorSpeedup = 1;
    if (targetHardware.accelerator && targetHardware.accelerator.type !== "None") {
      switch (targetHardware.accelerator.type) {
        case "GPU":
          acceleratorSpeedup = 3.5;
          break;
        case "TPU":
          acceleratorSpeedup = 5.0;
          break;
        case "NPU":
          acceleratorSpeedup = 4.0;
          break;
      }
    }
    
    // Simulate memory constraints impact
    // If model barely fits, it will be slower due to memory swapping
    const memoryPenalty = memoryConstraint < 0.8 ? (1 / memoryConstraint) * 2 : 1;
    
    // Calculate simulated inference time
    let inferenceTime = baseInferenceTime * cpuFactor / acceleratorSpeedup;
    inferenceTime *= memoryPenalty;
    
    // Power consumption based on processing time and hardware profile
    const basePowerUsage = targetHardware.powerConsumption;
    const powerFactor = acceleratorSpeedup > 1 ? 1.5 : 1; // Accelerators use more power
    const powerConsumption = basePowerUsage * powerFactor * (inferenceTime / baseInferenceTime);
    
    // Return simulated metrics
    return {
      size: modelSize,
      inferenceTime,
      memoryUsage,
      powerConsumption,
      accuracy: this.simulateAccuracyImpact(modelSize, inferenceTime, memoryConstraint)
    };
  }
  
  /**
   * Simulates the impact of hardware constraints on model accuracy
   */
  private static simulateAccuracyImpact(
    modelSize: number,
    inferenceTime: number,
    memoryConstraint: number
  ): number {
    // Base accuracy (simulated)
    const baseAccuracy = 90; // 90%
    
    // If model doesn't fit well in memory, accuracy might be affected
    const memoryImpact = memoryConstraint < 0.7 ? (memoryConstraint - 0.3) * 10 : 0;
    
    // Very small models might have lower accuracy due to over-optimization
    const sizeImpact = modelSize < 5 ? -5 : 0;
    
    // Return simulated accuracy with constraints applied
    return Math.min(100, Math.max(50, baseAccuracy + memoryImpact + sizeImpact));
  }
  
  /**
   * Simulates the maximum batch size the model can process on the hardware
   */
  static calculateMaxBatchSize(
    modelSize: number,
    targetHardware: TargetHardwareProfile
  ): number {
    // Calculate memory available for batching
    const runtimeOverhead = 200; // MB for runtime
    const memoryPerModel = modelSize * 1.2; // Each batch instance needs ~1.2x model size
    const availableMemory = targetHardware.memory - runtimeOverhead;
    
    // Calculate max batch size based on available memory
    return Math.max(1, Math.floor(availableMemory / memoryPerModel));
  }
  
  /**
   * Estimates battery life for the drone when using this model
   */
  static estimateBatteryLife(
    powerConsumption: number,
    batteryCapacity: number = 3500 // mAh
  ): number {
    // Assume standard drone battery voltage of 11.1V (3S LiPo)
    const batteryVoltage = 11.1;
    
    // Calculate battery capacity in watt-hours
    const batteryWattHours = (batteryCapacity / 1000) * batteryVoltage;
    
    // Estimate flight time in hours (assume model processing is 30% of power usage)
    const totalPowerConsumption = powerConsumption / 0.3; // Total drone power including flight
    
    // Return estimated battery life in minutes
    return (batteryWattHours / totalPowerConsumption) * 60;
  }
  
  /**
   * Simulates the thermal impact of running the model
   */
  static simulateThermalImpact(
    inferenceTime: number,
    powerConsumption: number
  ): {
    temperatureIncrease: number; // Celsius
    throttlingRisk: "low" | "medium" | "high";
  } {
    // Calculate heat generated based on power and time
    const heatFactor = 0.05; // Simplified conversion factor
    const temperatureIncrease = powerConsumption * heatFactor * (inferenceTime / 100);
    
    // Determine throttling risk based on temperature increase
    let throttlingRisk: "low" | "medium" | "high" = "low";
    if (temperatureIncrease > 20) {
      throttlingRisk = "high";
    } else if (temperatureIncrease > 10) {
      throttlingRisk = "medium";
    }
    
    return {
      temperatureIncrease,
      throttlingRisk
    };
  }
}

/**
 * Calculates flight endurance with the given model on a drone
 */
export function calculateFlightEndurance(
  modelMetrics: ModelMetrics,
  targetHardware: TargetHardwareProfile,
  droneSpecs: {
    batteryCapacity: number; // mAh
    weight: number; // grams
    efficiency: number; // g/W - how many grams it can lift per watt of power
  }
): {
  flightTime: number; // minutes
  framerate: number; // FPS
  realTimeDetection: boolean;
} {
  // Calculate power needed for flight
  const flightPower = droneSpecs.weight / droneSpecs.efficiency;
  
  // Power for the computing system
  const computePower = modelMetrics.powerConsumption || targetHardware.powerConsumption;
  
  // Total power consumption
  const totalPower = flightPower + computePower;
  
  // Battery capacity in watt-hours (assuming 11.1V battery)
  const batteryWattHours = (droneSpecs.batteryCapacity / 1000) * 11.1;
  
  // Flight time in hours
  const flightTimeHours = batteryWattHours / totalPower;
  
  // Framerate calculation (based on inference time)
  const framerate = 1000 / modelMetrics.inferenceTime;
  
  // Is it real-time? (Typically need at least 10 FPS for real-time detection)
  const realTimeDetection = framerate >= 10;
  
  return {
    flightTime: flightTimeHours * 60, // Convert to minutes
    framerate,
    realTimeDetection
  };
} 