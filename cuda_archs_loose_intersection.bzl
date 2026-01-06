def _parse_cuda_version(version_str):
    """
    Parses a CUDA arch string (e.g., "8.6", "9.0a") into a tuple for comparison.
    Returns: (major, minor, suffix_string)
    """
    if "." not in version_str:
        fail("Invalid CUDA arch format: " + version_str)
    
    parts = version_str.split(".")
    major = int(parts[0])
    rest = parts[1]
    
    # Separate minor version number from potential letter suffix (e.g. "0a" -> 0, "a")
    minor_digits = ""
    suffix = ""
    for i in range(len(rest)):
        char = rest[i]
        if char.isdigit():
            minor_digits += char
        else:
            suffix = rest[i:]
            break
            
    return (major, int(minor_digits), suffix)

def _version_le(v1, v2):
    """Returns True if v1 <= v2 based on semantic versioning."""
    return _parse_cuda_version(v1) <= _parse_cuda_version(v2)

def cuda_archs_loose_intersection(src_cuda_archs, tgt_cuda_archs):
    """
    Computes the loose intersection of CUDA architectures and formats them 
    as sm_XXX and compute_XXX strings.
    
    Args:
        src_cuda_archs: List of strings (e.g. ["7.5", "8.6", "9.0+PTX"])
        tgt_cuda_archs: List of strings (e.g. ["8.0", "9.0"])
        
    Returns:
        A List of strings formatted as ["sm_80", "sm_90", "compute_90", ...]
    """
    
    # Work with copies
    _src_list = [x for x in src_cuda_archs]
    _tgt_list = [x for x in tgt_cuda_archs]

    # 1. Handle +PTX suffix
    ptx_requests = {} # Set of base arches that requested PTX
    clean_src = []
    
    for arch in _src_list:
        if arch.endswith("+PTX"):
            base = arch[:-4] # Remove +PTX
            ptx_requests[base] = True
            clean_src.append(base)
        else:
            clean_src.append(arch)
    
    # Remove duplicates from source and sort
    _src_list = sorted(list(set(clean_src)), key=_parse_cuda_version)

    # 2. Special handling for x.0a (9.0a, 10.0a)
    result_list = []
    
    def _handle_special_case(special_ver, base_ver, src_list, tgt_list, res_list):
        if special_ver in src_list and base_ver in tgt_list:
            src_list.remove(special_ver)
            tgt_list.remove(base_ver)
            res_list.append(special_ver)
            return True
        return False

    _handle_special_case("9.0a", "9.0", _src_list, _tgt_list, result_list)
    _handle_special_case("10.0a", "10.0", _src_list, _tgt_list, result_list)

    # 3. Main Intersection Loop
    # Ensure source list is sorted for the logic
    _src_list = sorted(_src_list, key=_parse_cuda_version)
    
    for tgt in _tgt_list:
        tgt_parsed = _parse_cuda_version(tgt)
        tgt_major = tgt_parsed[0]
        
        best_match = None
        
        for src in _src_list:
            src_parsed = _parse_cuda_version(src)
            src_major = src_parsed[0]
            
            # Check version-less-or-equal
            if _version_le(src, tgt):
                # Check compatibility:
                # 1. If it was a PTX request, it matches across majors (JIT compatible)
                # 2. Otherwise, SASS requires matching Major version
                is_ptx_request = src in ptx_requests
                
                if is_ptx_request or (src_major == tgt_major):
                    best_match = src
            else:
                # Optimization: src list is sorted, so subsequent arches will be too high
                break
        
        if best_match:
            result_list.append(best_match)

    # 4. Post-processing and Formatting
    # Remove duplicates
    result_list = sorted(list(set(result_list)), key=_parse_cuda_version)
    
    final_output = []
    for arch in result_list:
        # Convert "8.0" -> "80", "9.0a" -> "90a"
        arch_code = arch.replace(".", "")
        
        # Always add binary architecture (SASS) -> sm_XXX
        final_output.append("sm_" + arch_code)
        
        # Add virtual architecture (PTX) if requested -> compute_XXX
        if arch in ptx_requests:
            final_output.append("compute_" + arch_code)
            
    return final_output
