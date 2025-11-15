# Scale-Adaptive Parameters for Event-Based Blade Tracking

## Overview

The current blade tracking implementation uses fixed pixel-based parameters that are optimized for a specific drone distance. This document outlines how to implement scale-adaptive parameters that automatically adjust based on the apparent size of the blades in the image, making the system robust to varying drone distances.

## Problem Statement

### Scale-Dependent Parameters

The following parameters are currently fixed in pixels and will be affected by changes in drone distance:

| Parameter | Current Value | Units | Impact When Drone Gets Closer | Impact When Drone Gets Farther |
|-----------|---------------|-------|-------------------------------|--------------------------------|
| `eps` | 2 | pixels | Too small - blades fragment | Too large - blades merge |
| `assignment_distance` | 10 | pixels | Too small - miss blade events | May work but suboptimal |
| `max_y_spread` | 5.0 | pixels | Too restrictive - reject valid blades | May work but suboptimal |
| `min_cluster_events` | 100 | count | May work (more events) | Too high - reject valid blades |

### Scale-Independent Parameters

These parameters are robust to distance changes:
- `window_us` (10ms) - Time-based
- `window_duration_us` (500μs) - Time-based  
- `min_samples` (5) - Relative count
- `grace_period_us` (20ms) - Time-based
- `reinit_interval_us` (10ms) - Time-based

## Proposed Solutions

### Option 1: Automatic Scale Detection

Detect the apparent scale of blades from the data itself and adjust parameters accordingly.

#### Implementation Steps:

1. **Estimate blade scale from initial clustering:**
   ```python
   def estimate_blade_scale(clusters):
       """Estimate average blade size in pixels from detected clusters."""
       blade_widths = []
       for cluster in clusters:
           x_values = cluster.events[:, 0]
           x_span = np.percentile(x_values, 95) - np.percentile(x_values, 5)
           blade_widths.append(x_span)
       
       return np.median(blade_widths) if blade_widths else 20.0  # Default: 20px
   ```

2. **Define baseline scale:**
   ```python
   # Reference scale (from our optimized tests at specific distance)
   BASELINE_SCALE = 20.0  # pixels (median blade width at reference distance)
   
   # Baseline parameters (optimized for BASELINE_SCALE)
   BASELINE_PARAMS = {
       'eps': 2,
       'assignment_distance': 10,
       'max_y_spread': 5.0,
       'min_cluster_events': 100
   }
   ```

3. **Scale parameters dynamically:**
   ```python
   def scale_parameters(detected_scale, baseline_scale=BASELINE_SCALE):
       """Scale pixel-based parameters based on detected blade size."""
       scale_factor = detected_scale / baseline_scale
       
       return {
           'eps': max(1, int(BASELINE_PARAMS['eps'] * scale_factor)),
           'assignment_distance': max(5, int(BASELINE_PARAMS['assignment_distance'] * scale_factor)),
           'max_y_spread': BASELINE_PARAMS['max_y_spread'] * scale_factor,
           'min_cluster_events': max(50, int(BASELINE_PARAMS['min_cluster_events'] / (scale_factor ** 0.5)))
       }
   ```

4. **Adaptive workflow:**
   ```
   1. Run initial clustering with baseline parameters
   2. Estimate blade scale from detected clusters
   3. Calculate scale_factor = detected_scale / baseline_scale
   4. Re-run clustering with scaled parameters if scale_factor > 1.5 or < 0.7
   5. Use scaled parameters for ongoing tracking
   ```

#### Pros:
- Fully automatic, no manual intervention
- Adapts to any distance
- Can re-adapt if drone moves during tracking

#### Cons:
- Requires good initial clustering to estimate scale
- Additional computational overhead
- May be unstable if initial clustering fails

---

### Option 2: Multi-Scale Parameter Sets

Pre-define parameter sets for different distance ranges and select based on initial conditions.

#### Implementation:

```python
PARAMETER_SETS = {
    'very_close': {  # < 1 meter
        'eps': 4,
        'assignment_distance': 20,
        'max_y_spread': 10.0,
        'min_cluster_events': 300,
        'expected_blade_width': (40, 80)  # pixels
    },
    'close': {  # 1-2 meters
        'eps': 3,
        'assignment_distance': 15,
        'max_y_spread': 7.0,
        'min_cluster_events': 150,
        'expected_blade_width': (25, 45)
    },
    'medium': {  # 2-4 meters (our current optimized range)
        'eps': 2,
        'assignment_distance': 10,
        'max_y_spread': 5.0,
        'min_cluster_events': 100,
        'expected_blade_width': (15, 30)
    },
    'far': {  # 4-6 meters
        'eps': 1,
        'assignment_distance': 7,
        'max_y_spread': 3.0,
        'min_cluster_events': 50,
        'expected_blade_width': (8, 18)
    },
    'very_far': {  # > 6 meters
        'eps': 1,
        'assignment_distance': 5,
        'max_y_spread': 2.0,
        'min_cluster_events': 25,
        'expected_blade_width': (3, 10)
    }
}

def select_parameter_set(initial_events):
    """Select best parameter set based on initial event analysis."""
    # Quick test with multiple parameter sets
    best_set = None
    best_score = -1
    
    for set_name, params in PARAMETER_SETS.items():
        clusters = test_clustering(initial_events, params)
        score = evaluate_clustering_quality(clusters, params)
        
        if score > best_score:
            best_score = score
            best_set = set_name
    
    return PARAMETER_SETS[best_set]

def evaluate_clustering_quality(clusters, params):
    """Score clustering quality based on expected characteristics."""
    if len(clusters) < 2 or len(clusters) > 6:
        return 0.0
    
    score = 0.0
    for cluster in clusters:
        x_width = cluster['x_span']
        
        # Check if blade width is in expected range
        min_width, max_width = params['expected_blade_width']
        if min_width <= x_width <= max_width:
            score += 1.0
        
        # Bonus for well-separated clusters
        score += 0.5 if cluster['y_spread'] < params['max_y_spread'] else 0.0
    
    return score / len(clusters)
```

#### Pros:
- Simple and predictable
- Pre-validated parameter sets
- Fast selection process
- Easy to add new distance ranges

#### Cons:
- Discrete jumps between parameter sets
- Requires manual calibration for each distance range
- May not handle intermediate distances optimally

---

### Option 3: Distance-Based Input

Allow user to specify approximate distance and scale parameters accordingly.

#### Implementation:

```yaml
# In config file
data:
  input_file: "drone_idle.dat"
  approximate_distance_meters: 3.0  # User-specified distance
  
scaling:
  enabled: true
  reference_distance_meters: 3.0  # Distance where baseline params were optimized
  reference_blade_width_pixels: 20.0
```

```python
def load_config_with_scaling(config_path, preset=None):
    """Load config and scale parameters based on distance."""
    config = load_config(config_path, preset)
    
    if config.get('scaling', {}).get('enabled', False):
        distance = config['data'].get('approximate_distance_meters', 3.0)
        ref_distance = config['scaling']['reference_distance_meters']
        
        # Scale inversely with distance (closer = larger in pixels)
        scale_factor = ref_distance / distance
        
        # Scale pixel-based parameters
        config['clustering']['initial']['eps'] = max(1, int(
            config['clustering']['initial']['eps'] * scale_factor
        ))
        config['clustering']['temporal']['assignment_distance'] = max(5, int(
            config['clustering']['temporal']['assignment_distance'] * scale_factor
        ))
        config['clustering']['initial']['max_y_spread'] = (
            config['clustering']['initial']['max_y_spread'] * scale_factor
        )
    
    return config
```

#### Pros:
- Simple to implement
- User has control
- Predictable behavior

#### Cons:
- Requires user to know/estimate distance
- Manual input needed
- Not adaptive if drone moves

---

### Option 4: Hybrid Approach (Recommended)

Combine automatic detection with parameter sets for robustness.

#### Implementation:

1. **Try clustering with baseline parameters**
2. **Estimate scale from results**
3. **If scale differs significantly, select appropriate parameter set**
4. **Re-cluster if needed**
5. **Use selected parameters for tracking**

```python
def adaptive_initialization(events, config):
    """Adaptively initialize clusters with scale-appropriate parameters."""
    
    # Try with baseline parameters first
    baseline_clusters = initialize_clusters(events, config['baseline_params'])
    
    if len(baseline_clusters) >= 2:
        # Estimate scale
        detected_scale = estimate_blade_scale(baseline_clusters)
        scale_factor = detected_scale / config['reference_scale']
        
        print(f"Detected blade scale: {detected_scale:.1f}px (factor: {scale_factor:.2f}x)")
        
        # If scale differs significantly, select better parameter set
        if scale_factor > 1.5:
            print("Switching to 'close' parameter set")
            params = PARAMETER_SETS['close']
        elif scale_factor < 0.7:
            print("Switching to 'far' parameter set")
            params = PARAMETER_SETS['far']
        else:
            print("Using baseline parameters")
            return baseline_clusters
        
        # Re-cluster with better parameters
        adapted_clusters = initialize_clusters(events, params)
        return adapted_clusters
    
    else:
        # Try multiple parameter sets if baseline failed
        print("Baseline clustering failed, trying parameter sets...")
        for set_name, params in PARAMETER_SETS.items():
            clusters = initialize_clusters(events, params)
            if 2 <= len(clusters) <= 6:
                print(f"Success with '{set_name}' parameter set")
                return clusters
        
        print("Warning: No parameter set produced good clustering")
        return baseline_clusters
```

#### Pros:
- Robust to various distances
- Automatic with fallback
- Best of both worlds
- Graceful degradation

#### Cons:
- More complex implementation
- Requires maintaining parameter sets
- Slight computational overhead

---

## Implementation Roadmap

### Phase 1: Infrastructure (1-2 days)
1. Add scale estimation function
2. Add parameter scaling functions
3. Add unit tests for scaling logic

### Phase 2: Parameter Sets (1 day)
1. Define 3-5 distance-based parameter sets
2. Add validation tests for each set
3. Document expected distance ranges

### Phase 3: Adaptive Logic (2-3 days)
1. Implement hybrid initialization
2. Add scale monitoring during tracking
3. Add optional re-adaptation if scale changes significantly

### Phase 4: Testing & Validation (2-3 days)
1. Test with drone at multiple distances
2. Validate RPM consistency across distances
3. Benchmark computational overhead
4. Document performance characteristics

### Phase 5: Configuration (1 day)
1. Add scaling options to config files
2. Add distance presets
3. Update documentation

---

## Testing Strategy

### Test Datasets Needed:
1. **Close range (1-2m):** Blades appear 40-80 pixels wide
2. **Medium range (2-4m):** Blades appear 15-30 pixels wide (current)
3. **Far range (4-6m):** Blades appear 8-18 pixels wide
4. **Variable range:** Drone moving closer/farther during recording

### Validation Metrics:
- **Clustering success rate:** Should detect 3-4 clusters at all distances
- **RPM consistency:** Same drone should give consistent RPM across distances
- **Computational cost:** Adaptive overhead should be < 10% of total time
- **Robustness:** Should handle 0.5x to 2.0x scale changes reliably

### Success Criteria:
- ✓ Detects correct number of blades at all test distances
- ✓ RPM estimates within ±10% across distances
- ✓ No manual parameter tuning needed for new distances
- ✓ Graceful degradation if scale detection fails

---

## Alternative: Deep Learning Approach

For production systems, consider training a neural network to:
1. Detect blade regions directly (scale-invariant)
2. Estimate blade parameters (width, rotation center, RPM)
3. Work at any distance without parameter tuning

**Pros:** Truly scale-invariant, can handle complex scenarios
**Cons:** Requires labeled training data, more complex deployment

---

## Recommendations

**For immediate use:** Keep current fixed parameters, document distance range (2-4 meters)

**For robustness:** Implement **Option 4 (Hybrid)** - provides best balance of automation and reliability

**For production:** Consider deep learning approach if dealing with highly variable scenarios

---

## Related Files

- `config/tracking_config.yaml` - Current configuration
- `scripts/test_clustering_robustness.py` - Parameter validation tool
- `scripts/test_rpm_robustness.py` - RPM consistency testing
- `scripts/create_tracking_video_configurable.py` - Main tracking implementation

---

## References

- Current optimized parameters determined from `test_clustering_robustness.py` results
- RPM stability validated with `test_rpm_robustness.py` (6.4% CV achieved)
- Scale dependency analysis based on DBSCAN algorithm characteristics