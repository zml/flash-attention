load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//:cuda_archs_loose_intersection.bzl", "cuda_archs_loose_intersection")

def _cuda_intersection_test_impl(ctx):
    env = unittest.begin(ctx)

    # Case 1: Simple Exact Match
    asserts.equals(
        env,
        ["sm_80"],
        cuda_archs_loose_intersection(["8.0"], ["8.0"])
    )

    # Case 2: PTX Request (8.0+PTX vs 9.0 target)
    # Should produce binary for 8.0 and virtual (compute) for 8.0
    asserts.equals(
        env,
        ["sm_80", "compute_80"],
        cuda_archs_loose_intersection(["8.0+PTX"], ["9.0"])
    )

    # Case 3: Special Version (9.0a vs 9.0 target)
    asserts.equals(
        env,
        ["sm_90a"],
        cuda_archs_loose_intersection(["9.0a"], ["9.0"])
    )

    # Case 4: Complex Intersection
    # 7.5 matches 7.5 (exact)
    # 8.0+PTX matches 8.0 (exact) -> adds compute
    # 9.0a matches 9.0 (special)
    asserts.equals(
        env,
        ["sm_75", "sm_80", "compute_80", "sm_90a"],
        cuda_archs_loose_intersection(
            ["7.5", "8.0+PTX", "9.0a"], 
            ["7.5", "8.0", "9.0"]
        )
    )
    
    asserts.equals(
        env,
        ["sm_80", "sm_86", "sm_90a"],
        cuda_archs_loose_intersection(
            ["7.5", "8.0", "8.6", "9.0", "9.0a"],
            ["8.0", "8.9", "9.0"]
        )
    )
    
    return unittest.end(env)

# Create the test rule
cuda_intersection_test = unittest.make(_cuda_intersection_test_impl)

def cuda_test_suite(name):
    unittest.suite(
        name,
        cuda_intersection_test,
    )
