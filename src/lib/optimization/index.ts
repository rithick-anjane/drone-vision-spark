// Export all types
export * from '@/lib/optimization/types';

// Export optimization services
export { OptimizationService } from '@/lib/optimization/optimizationService';
export { SimulationService, calculateFlightEndurance } from '@/lib/optimization/simulationService';
export { ModelFactory } from '@/lib/optimization/modelFactory';

// Export the core optimizer (though this will typically be used via the OptimizationService)
export { optimizeModel } from '@/lib/optimization/modelOptimizer'; 