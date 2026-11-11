// Gaussian splats, drawn as 3D Gaussian splatting draws them (splat.rs makes them): each one's 3D
// shape is projected to the 2D one it makes where it lands, drawn as a quad along that one's axes
// three standard deviations out, and laid over what's already there. They come sorted far to near.

struct Camera {
	// the turn, by columns
	r0: vec4<f32>,
	r1: vec4<f32>,
	r2: vec4<f32>,
	// what it turns round, and the focal length in pixels
	pivot: vec4<f32>,
	// the picture's width and height, the most a splat may spread (pixels squared), the near plane
	view: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct In {
	@builtin(vertex_index) corner: u32,
	// where it is, and how solid at its middle
	@location(0) a: vec4<f32>,
	// its two sizes, its angle, its thickness (a part of the smaller size)
	@location(1) b: vec4<f32>,
	@location(2) c: vec4<f32>,
};

struct Out {
	@builtin(position) at: vec4<f32>,
	// where in the splat, in standard deviations along its axes
	@location(0) local: vec2<f32>,
	@location(1) colour: vec4<f32>,
};

@vertex
fn vs(i: In) -> Out {
	var o: Out;
	// off the picture: not drawn
	o.at = vec4<f32>(2.0, 2.0, 2.0, 1.0);
	o.local = vec2<f32>(0.0, 0.0);
	o.colour = vec4<f32>(0.0, 0.0, 0.0, 0.0);
	let turn = mat3x3<f32>(cam.r0.xyz, cam.r1.xyz, cam.r2.xyz);
	let p = turn * (i.a.xyz - cam.pivot.xyz) + cam.pivot.xyz;
	if (p.z < cam.view.w || i.a.w <= 0.0) {
		return o;
	}
	// its shape: a disc facing the camera as it was made, turned in its own plane
	let ca = cos(i.b.z);
	let sa = sin(i.b.z);
	let thick = i.b.w * min(i.b.x, i.b.y);
	let m = mat3x3<f32>(vec3<f32>(ca, sa, 0.0) * i.b.x, vec3<f32>(-sa, ca, 0.0) * i.b.y, vec3<f32>(0.0, 0.0, thick));
	let rm = turn * m;
	// projected: the Jacobian of the perspective divide where it lands, in pixels
	let k = cam.pivot.w;
	let iz = 1.0 / p.z;
	let j0 = vec3<f32>(k * iz, 0.0, -k * p.x * iz * iz);
	let j1 = vec3<f32>(0.0, k * iz, -k * p.y * iz * iz);
	let t0 = vec3<f32>(dot(j0, rm[0]), dot(j0, rm[1]), dot(j0, rm[2]));
	let t1 = vec3<f32>(dot(j1, rm[0]), dot(j1, rm[1]), dot(j1, rm[2]));
	// its 2D covariance, a little blurred so the smallest don't flicker
	let sxx = dot(t0, t0) + 0.3;
	let sxy = dot(t0, t1);
	let syy = dot(t1, t1) + 0.3;
	// its axes: the covariance's eigenvectors, no longer than the picture allows
	let mid = 0.5 * (sxx + syy);
	let d = sqrt(max(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy, 0.0));
	let l1 = min(mid + d, cam.view.z);
	let l2 = clamp(mid - d, 0.01, l1);
	var e1 = vec2<f32>(1.0, 0.0);
	if (abs(sxy) > 1e-6) {
		e1 = normalize(vec2<f32>(sxy, mid + d - sxx));
	} else if (syy > sxx) {
		e1 = vec2<f32>(0.0, 1.0);
	}
	let e2 = vec2<f32>(-e1.y, e1.x);
	let corner = vec2<f32>(f32(i.corner & 1u), f32(i.corner >> 1u)) * 2.0 - 1.0;
	let centre = vec2<f32>(cam.view.x * 0.5 + k * p.x * iz, cam.view.y * 0.5 + k * p.y * iz);
	let px = centre + 3.0 * (corner.x * sqrt(l1) * e1 + corner.y * sqrt(l2) * e2);
	o.at = vec4<f32>(px.x / cam.view.x * 2.0 - 1.0, 1.0 - px.y / cam.view.y * 2.0, 0.0, 1.0);
	o.local = corner * 3.0;
	o.colour = vec4<f32>(i.c.rgb, i.a.w);
	return o;
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
	let a = min(o.colour.a * exp(-0.5 * dot(o.local, o.local)), 0.99);
	if (a < 1.0 / 255.0) {
		discard;
	}
	return vec4<f32>(o.colour.rgb * a, a);
}
