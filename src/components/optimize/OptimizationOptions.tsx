import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { 
  OptimizationConfig,
  OptimizationService,
  PruningType,
  QuantizationType,
  HARDWARE_PROFILES
} from "@/lib/optimization";

interface OptimizationOptionsProps {
  onStartOptimization: (config: OptimizationConfig) => void;
  disabled?: boolean;
}

const OptimizationOptions = ({ onStartOptimization, disabled = false }: OptimizationOptionsProps) => {
  // Get optimization service instance
  const optimizationService = OptimizationService.getInstance();
  
  // Default optimization config
  const [config, setConfig] = useState<OptimizationConfig>({
    pruningEnabled: true,
    pruningType: "structured",
    compressionLevel: 50,
    quantizationEnabled: true,
    quantizationType: "post-training",
    quantizationPrecision: 8,
    fineTuningEnabled: true,
    fineTuningSteps: 100,
    targetHardware: HARDWARE_PROFILES["generic-drone"]
  });
  
  // Template configurations
  const [selectedPreset, setSelectedPreset] = useState<string>("none");
  
  // Update config when making changes
  const updateConfig = <K extends keyof OptimizationConfig>(key: K, value: OptimizationConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setSelectedPreset("none"); // Reset preset when making manual changes
  };
  
  // Load preset configuration
  const loadPreset = (presetId: string) => {
    if (presetId === "none") return;
    
    const presets = optimizationService.getExampleConfigurations();
    const preset = presets[presetId];
    
    if (preset) {
      setConfig(preset);
      setSelectedPreset(presetId);
    }
  };
  
  // Handle form submission
  const handleOptimize = () => {
    onStartOptimization(config);
  };

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Optimization Options</CardTitle>
        <div className="flex items-center gap-2">
          <Label htmlFor="preset">Preset</Label>
          <Select value={selectedPreset} onValueChange={loadPreset}>
            <SelectTrigger id="preset" className="w-[180px]">
              <SelectValue placeholder="Select preset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">Custom Settings</SelectItem>
              <SelectItem value="balanced">Balanced</SelectItem>
              <SelectItem value="max-speed">Maximum Speed</SelectItem>
              <SelectItem value="max-accuracy">Maximum Accuracy</SelectItem>
              <SelectItem value="tpu-optimized">TPU Optimized</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="methods" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="methods">Methods</TabsTrigger>
            <TabsTrigger value="hardware">Hardware</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>
          
          <TabsContent value="methods" className="space-y-4 py-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="pruning" className="text-base">Pruning</Label>
                <p className="text-sm text-muted-foreground">
                  Remove redundant weights from your model
                </p>
              </div>
              <Switch
                id="pruning"
                checked={config.pruningEnabled}
                onCheckedChange={(checked) => updateConfig("pruningEnabled", checked)}
                disabled={disabled}
              />
            </div>
            
            {config.pruningEnabled && (
              <div className="pl-4 border-l-2 border-muted-foreground/20 space-y-4 mt-2">
                <div className="space-y-2">
                  <Label htmlFor="pruning-type">Pruning Method</Label>
                  <Select
                    value={config.pruningType}
                    onValueChange={(value) => updateConfig("pruningType", value as PruningType)}
                    disabled={disabled}
                  >
                    <SelectTrigger id="pruning-type">
                      <SelectValue placeholder="Select pruning method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="unstructured">
                        <div className="flex flex-col">
                          <span>Unstructured Pruning</span>
                          <span className="text-xs text-muted-foreground">
                            Remove individual weights (best for accuracy)
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="structured">
                        <div className="flex flex-col">
                          <span>Structured Pruning</span>
                          <span className="text-xs text-muted-foreground">
                            Remove entire channels (best for speed)
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="group-channel">
                        <div className="flex flex-col">
                          <span>Group Channel Pruning</span>
                          <span className="text-xs text-muted-foreground">
                            Chu et al. method for optimal balance
                          </span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label htmlFor="compression-level">Pruning Intensity</Label>
                    <span className="text-sm text-muted-foreground w-12 text-right">
                      {config.compressionLevel}%
                    </span>
                  </div>
                  <Slider
                    id="compression-level"
                    value={[config.compressionLevel]}
                    onValueChange={(value) => updateConfig("compressionLevel", value[0])}
                    min={10}
                    max={90}
                    step={5}
                    disabled={disabled}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground pt-1">
                    <span>Conservative</span>
                    <span>Aggressive</span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-between mt-6">
              <div>
                <Label htmlFor="quantization" className="text-base">Quantization</Label>
                <p className="text-sm text-muted-foreground">
                  Reduce numerical precision of weights
                </p>
              </div>
              <Switch
                id="quantization"
                checked={config.quantizationEnabled}
                onCheckedChange={(checked) => updateConfig("quantizationEnabled", checked)}
                disabled={disabled}
              />
            </div>
            
            {config.quantizationEnabled && (
              <div className="pl-4 border-l-2 border-muted-foreground/20 space-y-4 mt-2">
                <div className="space-y-2">
                  <Label htmlFor="quantization-type">Quantization Method</Label>
                  <Select
                    value={config.quantizationType}
                    onValueChange={(value) => updateConfig("quantizationType", value as QuantizationType)}
                    disabled={disabled}
                  >
                    <SelectTrigger id="quantization-type">
                      <SelectValue placeholder="Select quantization method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="post-training">
                        <div className="flex flex-col">
                          <span>Post-Training Quantization</span>
                          <span className="text-xs text-muted-foreground">
                            Convert weights after training (fastest)
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="quantization-aware">
                        <div className="flex flex-col">
                          <span>Quantization-Aware Training</span>
                          <span className="text-xs text-muted-foreground">
                            Simulate quantization during training (better accuracy)
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="optimal-brain">
                        <div className="flex flex-col">
                          <span>Optimal Brain Quantizer</span>
                          <span className="text-xs text-muted-foreground">
                            Advanced method using Hessian approximation
                          </span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="precision">Precision</Label>
                  <Select
                    value={config.quantizationPrecision.toString()}
                    onValueChange={(value) => updateConfig("quantizationPrecision", parseInt(value) as 8 | 16 | 32)}
                    disabled={disabled}
                  >
                    <SelectTrigger id="precision">
                      <SelectValue placeholder="Select precision" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="8">
                        <div className="flex flex-col">
                          <span>8-bit Integer (INT8)</span>
                          <span className="text-xs text-muted-foreground">
                            Best for efficiency, may impact accuracy
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="16">
                        <div className="flex flex-col">
                          <span>16-bit Float (FP16)</span>
                          <span className="text-xs text-muted-foreground">
                            Good balance of efficiency and accuracy
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="32">
                        <div className="flex flex-col">
                          <span>32-bit Float (FP32)</span>
                          <span className="text-xs text-muted-foreground">
                            Full precision, no quantization
                          </span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="hardware" className="py-4">
            <div className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="target-hardware">Target Hardware</Label>
                <Select
                  value={Object.keys(HARDWARE_PROFILES).find(
                    key => HARDWARE_PROFILES[key].name === config.targetHardware.name
                  ) || "generic-drone"}
                  onValueChange={(value) => updateConfig("targetHardware", HARDWARE_PROFILES[value])}
                  disabled={disabled}
                >
                  <SelectTrigger id="target-hardware">
                    <SelectValue placeholder="Select target hardware" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="generic-drone">Generic Drone</SelectItem>
                    <SelectItem value="raspberry-pi">Raspberry Pi 4</SelectItem>
                    <SelectItem value="jetson-nano">NVIDIA Jetson Nano</SelectItem>
                    <SelectItem value="intel-ncs">Intel Neural Compute Stick 2</SelectItem>
                    <SelectItem value="coral-tpu">Google Coral TPU</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2 bg-muted/50 p-4 rounded-lg">
                <h3 className="font-medium">Hardware Specifications</h3>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div>
                    <p className="text-sm font-medium">CPU</p>
                    <p className="text-xs text-muted-foreground">
                      {config.targetHardware.cpu.cores} cores @ {config.targetHardware.cpu.frequency} MHz
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Memory</p>
                    <p className="text-xs text-muted-foreground">
                      {config.targetHardware.memory} MB
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Power</p>
                    <p className="text-xs text-muted-foreground">
                      {config.targetHardware.powerConsumption} Watts
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Accelerator</p>
                    <p className="text-xs text-muted-foreground">
                      {config.targetHardware.accelerator ? 
                        `${config.targetHardware.accelerator.type} (${config.targetHardware.accelerator.memory} MB)` : 
                        "None"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="advanced" className="py-4">
            <div className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="export-format">Export Format</Label>
                <Select
                  value={config.exportFormat || "same"}
                  onValueChange={(value) => updateConfig("exportFormat", value === "same" ? undefined : value)}
                  disabled={disabled}
                >
                  <SelectTrigger id="export-format">
                    <SelectValue placeholder="Select export format" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="same">Same as Input</SelectItem>
                    <SelectItem value="pt">PyTorch (.pt)</SelectItem>
                    <SelectItem value="onnx">ONNX (.onnx)</SelectItem>
                    <SelectItem value="tflite">TensorFlow Lite (.tflite)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="fine-tuning" className="text-base">Fine-tuning Pass</Label>
                  <p className="text-sm text-muted-foreground">
                    Run a fine-tuning pass to recover accuracy
                  </p>
                </div>
                <Switch 
                  id="fine-tuning" 
                  checked={config.fineTuningEnabled}
                  onCheckedChange={(checked) => updateConfig("fineTuningEnabled", checked)}
                  disabled={disabled}
                />
              </div>
              
              {config.fineTuningEnabled && (
                <div className="pl-4 border-l-2 border-muted-foreground/20 space-y-4 mt-2">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="fine-tuning-steps">Fine-tuning Steps</Label>
                      <span className="text-sm text-muted-foreground w-12 text-right">
                        {config.fineTuningSteps}
                      </span>
                    </div>
                    <Slider
                      id="fine-tuning-steps"
                      value={[config.fineTuningSteps]}
                      onValueChange={(value) => updateConfig("fineTuningSteps", value[0])}
                      min={10}
                      max={300}
                      step={10}
                      disabled={disabled}
                    />
                    <div className="flex justify-between text-xs text-muted-foreground pt-1">
                      <span>Fast</span>
                      <span>Thorough</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="mt-6">
          <Button 
            className="w-full" 
            size="lg" 
            onClick={handleOptimize}
            disabled={disabled}
          >
            Optimize Model
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default OptimizationOptions;
