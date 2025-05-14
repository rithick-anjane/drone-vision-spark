import { useState, useEffect } from "react";
import PageLayout from "@/components/layout/PageLayout";
import ModelUploader from "@/components/optimize/ModelUploader";
import OptimizationOptions from "@/components/optimize/OptimizationOptions";
import OptimizationStatus from "@/components/optimize/OptimizationStatus";
import ModelComparison from "@/components/optimize/ModelComparison";
import { OptimizationService, OptimizationConfig, OptimizationResult } from "@/lib/optimization";

const Optimize = () => {
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<OptimizationResult | null>(null);
  
  // Get optimization service instance
  const optimizationService = OptimizationService.getInstance();
  
  // Set up progress tracking when component mounts
  useEffect(() => {
    const unsubscribe = optimizationService.onProgress(handleProgressUpdate);
    return () => unsubscribe();
  }, []);
  
  // Handle progress updates
  const handleProgressUpdate = (newProgress: number) => {
    setProgress(newProgress);
    
    // Mark as complete when progress reaches 100%
    if (newProgress >= 100) {
      setIsComplete(true);
      setIsOptimizing(false);
    }
  };
  
  // Handle starting the optimization
  const handleStartOptimization = async (config: OptimizationConfig) => {
    try {
      setIsOptimizing(true);
      setIsComplete(false);
      setProgress(0);
      
      // Start the optimization process
      const optimizationResult = await optimizationService.optimizeCurrentModel(config);
      
      // Set the result
      setResult(optimizationResult);
      
      // This is handled by the progress callback, but setting explicitly just in case
      setIsComplete(true);
      setIsOptimizing(false);
    } catch (error) {
      console.error("Optimization failed:", error);
      setIsOptimizing(false);
    }
  };
  
  return (
    <PageLayout>
      <div className="container px-4 md:px-6 py-12 space-y-10">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Optimize Your Model</h1>
          <p className="mt-2 text-lg text-muted-foreground">
            Upload and optimize your computer vision model for drone deployment
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <ModelUploader />
          <OptimizationOptions 
            onStartOptimization={handleStartOptimization}
            disabled={isOptimizing}
          />
        </div>
        
        <OptimizationStatus 
          isOptimizing={isOptimizing} 
          progress={progress} 
          isComplete={isComplete} 
        />
        
        {isComplete && result && <ModelComparison result={result} />}
      </div>
    </PageLayout>
  );
};

export default Optimize;
