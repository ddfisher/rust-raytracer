// TODO: remove once stable
#![allow(dead_code)]

extern crate image;

use std::vec::Vec;
use std::f32::consts;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::iter;
use std::ops::{Add, Sub, Mul};

// TODO: support triangles
// TODO: parse/import .obj files
// TODO: support .mtl files
// TODO: speed up ray intersection with KD trees or similar
// TODO: antialiasing: http://en.wikipedia.org/wiki/Supersampling
// TODO: consider making multithreaded
// TODO: make GUI where you can move camera around
// TODO: refactor?


// TODO: consider using a more fine grained Color for computation
#[derive(Debug, Copy, Clone)]
struct Color {
    red: u8,
    green: u8,
    blue: u8
}

impl Add for Color {
    type Output = Color;

    fn add(self, other: Color) -> Color {
        Color {
            red:   self.red.saturating_add(other.red),
            green: self.green.saturating_add(other.green),
            blue:  self.blue.saturating_add(other.blue)
        }
    }
}

impl Sub for Color {
    type Output = Color;

    fn sub(self, other: Color) -> Color {
        Color {
            red:   self.red.saturating_sub(other.red),
            green: self.green.saturating_sub(other.green),
            blue:  self.blue.saturating_sub(other.blue)
        }
    }
}

impl Mul<f32> for Color {
    type Output = Color;

    fn mul(self, f: f32) -> Color {
        fn mul_sat(n: u8, f: f32) -> u8 {
            let p = n as f32 * f;
            let max: u8 = u8::max_value();
            if p < (max as f32) { p as u8 } else { 0xFF }
        }

        Color {
            red:   mul_sat(self.red,   f),
            green: mul_sat(self.green, f),
            blue:  mul_sat(self.blue,  f)
        }
    }
}

struct Image {
    width: u32,
    height: u32,
    pixels: Vec<Color>
}

impl Color {
    fn new(r: u8, g: u8, b: u8) -> Color {
        Color { red: r, green: g, blue: b }
    }

    fn black() -> Color {
        Color {red: 0, green: 0, blue: 0}
    }

    fn white() -> Color {
        Color {red: 0xFF, green: 0xFF, blue: 0xFF}
    }

    fn bytes(&self) -> Vec<u8> {
        vec![self.red, self.green, self.blue]
    }
}

impl Image {
    fn new(width: u32, height: u32) -> Image {
        Image {
            width: width,
            height: height,
            pixels: iter::repeat(Color::black()).take((width * height) as usize).collect()
        }
    }

    // TODO: try to remove some copying?
    fn to_pixels(&self) -> Vec<u8> {
        let mut pixel_vector: Vec<u8> = Vec::with_capacity((self.width * self.height * 3) as usize);
        for pixel in self.pixels.iter() {
            pixel_vector.push(pixel.red);
            pixel_vector.push(pixel.green);
            pixel_vector.push(pixel.blue);
        }
        pixel_vector
    }
}

fn main() {
    let angles = 64u32;
    // for i in (0u32..angles) {
    for i in 0u32..1 { // XXX
        let pathname = format!("scene{:02}.png", i);
        let path = Path::new(&pathname);
        let image = raytrace(consts::PI * 2.0 * (i as f32 / angles as f32));
        // let mut png = image.to_png_image();

        // png::store_png(&mut png, &path).unwrap();
        image::save_buffer(&path, &image.to_pixels()[..], image.width, image.height, image::RGB(8)).unwrap()
    }
}

fn raytrace(orientation: f32) -> Image {
    let width = 900; // XXX
    let height = 900;
    let mut img = Image::new(width, height);
    let scene = setup_scene();

    for x in 0..width {
        for y in 0..height {
            let ray = pixel_to_ray(x, y, width, height, orientation);
            let pixel = ray_to_color(&ray, &scene, 0);
            img.pixels[(x + y * width) as usize] = pixel;
        }
    }

    img
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
    z: f32
}

#[derive(Debug, Clone, Copy)]
struct Vector {
    dx: f32,
    dy: f32,
    dz: f32
}

impl Point {
    fn distance(self, other: Point) -> f32 {
        (self - other).length()
    }
}

impl Vector {
    fn length(&self) -> f32 {
        f32::sqrt(self.dx * self.dx + self.dy * self.dy + self.dz * self.dz)
    }

    fn normalized(&self) -> Vector {
        let length = self.length();
        Vector {
            dx: self.dx / length,
            dy: self.dy / length,
            dz: self.dz / length
        }
    }

    fn dot(&self, other: &Vector) -> f32 {
        self.dx * other.dx + self.dy * other.dy + self.dz * other.dz
    }

    fn cross(&self, other: &Vector) -> Vector {
        Vector {
            dx: self.dy * other.dz - self.dz * other.dy,
            dy: self.dz * other.dx - self.dx * other.dz,
            dz: self.dx * other.dy - self.dy * other.dx
        }
    }

    fn times(&self, scalar: f32) -> Vector {
        Vector {
            dx: self.dx * scalar,
            dy: self.dy * scalar,
            dz: self.dz * scalar
        }
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        Vector {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            dz: self.dz + other.dz
        }
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, other: Vector) -> Vector {
        Vector {
            dx: self.dx - other.dx,
            dy: self.dy - other.dy,
            dz: self.dz - other.dz
        }
    }
}

impl Add<Vector> for Point {
    type Output = Point;

    fn add(self, rel: Vector) -> Point {
        Point {
            x: self.x + rel.dx,
            y: self.y + rel.dy,
            z: self.z + rel.dz
        }
    }
}

impl Sub<Point> for Point {
    type Output = Vector;

    fn sub(self, other: Point) -> Vector {
        Vector {
            dx: self.x - other.x,
            dy: self.y - other.y,
            dz: self.z - other.z
        }
    }
}


#[derive(PartialEq, PartialOrd)]
struct OrderedF32(f32);

impl Eq for OrderedF32{} // because I'm a bad person

impl Ord for OrderedF32 {
    fn cmp(&self, &OrderedF32(other): &OrderedF32) -> Ordering {
        let &OrderedF32(s) = self;
        match (s.is_nan(), other.is_nan()) {
            (true, true)   => Ordering::Equal,
            (true, false)  => Ordering::Less,
            (false, true)  => Ordering::Greater,
            (false, false) => if s == other { Ordering::Equal } else if s < other { Ordering::Less } else { Ordering::Greater }
        }
    }
}

// goes from p0 towards p1
struct Ray {
    origin: Point,
    direction: Vector
}

struct SceneObject {
    shape: Shape,
    properties: MaterialProperties
}


enum Shape {
    Sphere {
        center: Point,
        radius: f32
    },
    Plane {
        point: Point,
        normal: Vector
    },
    Triangle {
        p1: Point,
        p2: Point,
        p3: Point
    }
}

struct MaterialProperties {
    color_primary: Color,
    color_secondary: Color,
    specular: f32,
    diffuse: f32,
    ambient: f32,
    shininess: f32,
    reflectivity: f32
}

// TODO: consider other values of epsilon
static EPSILON: f32 = 0.000001;
static SELF_INTERSECT_OFFSET: f32 = 0.0001;
static MAX_BOUNCES: u32 = 1;

impl SceneObject {
    // TODO: consider refactoring to return distance instead of intersection point
    fn hit(&self, &Ray{origin: ref o, direction: ref d}: &Ray) -> Option<Point> {
        // TODO: optimize
        match self.shape {
            Shape::Sphere {ref center, ref radius} => {
                // http://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
                let offset = *o - *center;
                let initial = -d.dot(&offset);
                let discriminant = d.dot(&offset).powi(2) - offset.length().powi(2) + radius.powi(2);
                if discriminant < 0.0 {
                    None
                } else {
                    let radical = discriminant.sqrt();
                    let farther_distance = initial + radical;
                    let closer_distance = initial - radical;
                    if farther_distance < 0.0 {
                        // both intersection points are behind the ray
                        None
                    } else if closer_distance < 0.0 {
                        // the closer intersection point is behind the ray, so use the farther one
                        Some(farther_distance)
                    } else {
                        Some(closer_distance)
                    }.map(|dist| *o + d.times(dist))
                }
            },
            Shape::Plane {ref point, ref normal} => {
                // http://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
                let offset = (*point - *o).dot(normal);
                let divergence = d.dot(normal);
                if divergence.abs() < EPSILON {
                    // line is parallel to plane
                    if offset.abs() < EPSILON {
                        // ray intersects plane everywhere, so we return the start of the ray
                        Some(o.clone())
                    } else {
                        // ray does not intersect plane
                        None
                    }
                } else {
                    let distance = offset / divergence;
                    if distance <= 0.0 {
                        // intersection point is before the start of the ray
                        None
                    } else {
                        // intersects plane at one point
                        Some(*o + d.times(distance))
                    }
                }
            },
            Shape::Triangle {ref p1, ref p2, ref p3} => {
                // http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
                // Find vectors for two edges sharing P1
                let e1 = *p2 - *p1;
                let e2 = *p3 - *p1;
                // Begin calculating determinant - also used to calculate u parameter
                let p = d.cross(&e2);
                // if determinant is near zero, ray lies in plane of triangle
                let det = e1.dot(&p);
                if det.abs() < EPSILON { return None; }
                let inv_det = 1.0 / det;

                // Calculate distance from P1 to ray origin
                let p1_dist = *o - *p1;

                // Calculate u parameter and test bound
                let u = p1_dist.dot(&p) * inv_det;
                // The intersection lies outside of the triangle
                if u < 0.0 || u > 1.0 { return None; }

                // Prepare to test v parameter
                let q = p1_dist.cross(&e1);
                // Calculate V parameter and test bound
                let v = d.dot(&q) * inv_det;
                // The intersection lies outside of the triangle
                if v < 0.0 || u + v  > 1.0 { return None; }

                let t = e2.dot(&q) * inv_det;

                if t > EPSILON { // ray intersection
                    return Some(*o + d.times(t));
                }

                // No hit
                return None;
            }
        }
    }

    fn normal_at(&self, point: &Point) -> Vector {
        match self.shape {
            Shape::Sphere {ref center, ..} => {
                (*point - *center).normalized()
            },
            Shape::Plane {ref normal, ..} => normal.clone(),
            Shape::Triangle {ref p1, ref p2, ref p3} => {
                let e1 = *p2 - *p1;
                let e2 = *p3 - *p1;
                e1.cross(&e2).normalized()
            }
        }
    }

    fn reflection_at(&self, point: &Point, direction: &Vector) -> Vector {
        let norm_v = self.normal_at(point);
        *direction - norm_v.times(2.0 * norm_v.dot(direction))
    }

    // TODO: set color function in object instead?
    fn color_at(&self, point: &Point) -> &Color {
        match self.shape {
            Shape::Sphere {..} => &self.properties.color_primary,
            Shape::Plane {..} => {
                // TODO: make this work for non-horizontal grids
                let grid_size = 0.5;
                let x = (point.x < 0.0) ^ (point.x.abs() % (grid_size * 2.0) < grid_size);
                let y = (point.z < 0.0) ^ (point.z.abs() % (grid_size * 2.0) < grid_size);
                if x ^ y {
                    &self.properties.color_primary
                } else {
                    &self.properties.color_secondary
                }
            },
            Shape::Triangle {..} => &self.properties.color_primary
        }
    }
}

struct Scene {
    objects: Vec<SceneObject>,
    lights: Vec<Light>
}

impl Scene {
    fn hit(&self, ray: &Ray) -> Option<(Point, &SceneObject)> {
        self.objects.iter()
            .filter_map(|obj| obj.hit(ray).map(|p| (p,obj)))
            .min_by_key(|&(ref p,_)| OrderedF32(p.distance(ray.origin)))
    }

    fn hit_any(&self, ray: &Ray, ignore_obj: &SceneObject) -> bool {
        self.objects.iter()
            .filter(|&obj| (obj as *const SceneObject) != (ignore_obj as *const SceneObject)) // pointer equality
            .any(|obj| obj.hit(ray).is_some())
    }
}

enum Light {
    // Point {
    //     point: Point,
    //     intensity: f32
    // },
    Direction {
        direction: Vector,
        intensity: f32
    }
}

impl Light {
    fn vector_for(&self, _: &Point) -> Vector {
        match self {
            &Light::Direction {ref direction, ..} => direction.normalized().times(-1.0)
        }
    }

    fn intensity_for(&self, _: &Point) -> f32 {
        match self {
            &Light::Direction {intensity, ..} => intensity
        }
    }
}

// TODO: fix up the projection
fn pixel_to_ray(x: u32, y: u32, width: u32, height: u32, orientation: f32) -> Ray {
    let camera = Point {
        x: 7.0 * orientation.sin(),
        y: 0.5,
        z: -7.0 * orientation.cos()
    };
    let fov_spread = 1.0;
    Ray {
        direction:
            (Point {
            x: (x as f32 / width as f32 * fov_spread - fov_spread / 2.0) * orientation.cos() + 6.0 * orientation.sin(),
            y: (height - y) as f32 / height as f32 * fov_spread - fov_spread / 2.0 + 0.5,
            z: (x as f32 / width as f32 * fov_spread - fov_spread / 2.0) * orientation.sin() - 6.0 * orientation.cos()
        } - camera).normalized(),
        origin: camera
    }
}

fn setup_scene() -> Scene {
    // let teapot = parse_simple_obj_file("scene/teapot.obj");
    let teapot = parse_simple_obj_file("scene/dodecahedron.obj");
    let mut objs = vec![
            // SceneObject {
            //     shape: Shape::Sphere {
            //         center: Point {x:0.0, y:0.0, z:0.0},
            //         radius: 1.0
            //     },
            //     properties: MaterialProperties {
            //         color_primary: Color {
            //             red: 0xFF,
            //             green: 0x00,
            //             blue: 0x00
            //         },
            //         color_secondary: Color::black(),
            //         specular: 1.0,
            //         diffuse: 0.8,
            //         ambient: 0.2,
            //         shininess: 13.0,
            //         reflectivity: 0.5
            //     }
            // },
            SceneObject {
                shape: Shape::Sphere {
                    center: Point {x:3.0, y:0.0, z:0.0},
                    radius: 0.5
                },
                properties: MaterialProperties {
                    color_primary: Color {
                        red: 0x00,
                        green: 0xFF,
                        blue: 0x00
                    },
                    color_secondary: Color::black(),
                    specular: 1.0,
                    diffuse: 0.8,
                    ambient: 0.2,
                    shininess: 13.0,
                    reflectivity: 0.5
                }
            },
            SceneObject {
                shape: Shape::Plane {
                    point: Point {x:0.0, y:-2.0, z:0.0},
                    normal: Vector {dx: 0.0, dy: 1.0, dz:0.0}
                },
                properties: MaterialProperties {
                    color_primary: Color {
                        red: 0x50,
                        green: 0x30,
                        blue: 0xA0
                    },
                    color_secondary: Color::white(),
                    specular: 0.0,
                    diffuse: 1.0,
                    ambient: 0.2,
                    shininess: 2.0,
                    reflectivity: 0.0
                }
            },
            // SceneObject {
            //     shape: Shape::Triangle {
            //         p1: Point {x:0.0, y:0.0, z:0.5},
            //         p2: Point {x:1.0, y:1.0, z:0.0},
            //         p3: Point {x:1.0, y:0.0, z:0.0},
            //     },
            //     properties: MaterialProperties {
            //         color_primary: Color {
            //             red: 0x00,
            //             green: 0x00,
            //             blue: 0xFF
            //         },
            //         color_secondary: Color::black(),
            //         specular: 1.0,
            //         diffuse: 0.8,
            //         ambient: 0.2,
            //         shininess: 13.0,
            //         reflectivity: 0.0
            //     }
            // },
        ];
    objs.extend(teapot.into_iter());
    Scene {
        objects: objs,
        lights: vec![
            Light::Direction {
                direction: Vector {
                    dx: -2.0,
                    dy: -1.0,
                    dz: 0.0
                },
                intensity: 0.2
            },
            Light::Direction {
                direction: Vector {
                    dx: 1.5,
                    dy: -3.0,
                    dz: 1.0
                },
                intensity: 0.8
            }
            // Light::Direction {
            //     direction: Vector {
            //         dx: 0.0,
            //         dy: 0.0,
            //         dz: 1.0
            //     },
            //     intensity: 0.8
            // }
        ]
    }
}

fn ray_to_color(ray: &Ray, scene: &Scene, bounces: u32) -> Color {
    fn light_contribution(light: &Light, point: &Point, obj: &SceneObject, ray: &Ray) -> Color {
        // http://en.wikipedia.org/wiki/Lambertian_reflectance
        let norm_v = obj.normal_at(point);
        let light_v = &light.vector_for(point);
        let diffuse_intensity = obj.properties.diffuse * f32::max(0.0, norm_v.dot(light_v));
        // http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model
        let half_way_v = (ray.direction.times(-1.0) + *light_v).normalized();
        let specular_intensity = obj.properties.specular
            * f32::max(0.0, norm_v.dot(&half_way_v)).powf(obj.properties.shininess);
        let diffuse_color = *obj.color_at(point) * diffuse_intensity;
        let specular_color = (Color::white() - diffuse_color) * specular_intensity;
        (diffuse_color + specular_color) * light.intensity_for(point)
    }


    // TODO: remove duplicate normal calls
    match scene.hit(ray) {
        Some((point, obj)) => {
            let ambient = *obj.color_at(&point) * obj.properties.ambient;
            let local_color = ambient + scene.lights.iter()
                .filter(|l| {
                    let purturbed_point = point + obj.normal_at(&point).times(SELF_INTERSECT_OFFSET);
                    !scene.hit_any(&Ray {direction: l.vector_for(&purturbed_point), origin: purturbed_point}, obj)
                })
                .map(|l| light_contribution(l, &point, obj, ray))
                .fold(Color::black(), |a, b| a + b);
            if bounces >= MAX_BOUNCES {
                local_color
            } else {
                let purturbed_point = point + obj.normal_at(&point).times(SELF_INTERSECT_OFFSET);
                let reflected_ray = Ray {
                    origin: purturbed_point,
                    direction: obj.reflection_at(&point, &ray.direction)
                };
                let reflection_color = ray_to_color(&reflected_ray, scene, bounces + 1);
                reflection_color * obj.properties.reflectivity + local_color// * (1.0 - obj.properties.reflectivity) // XXX
            }
        },
        None => Color::black()
    }
}

fn parse_simple_obj_file(filename: &str) -> Vec<SceneObject> {
    let path = Path::new(filename);
    let file = BufReader::new(File::open(&path).unwrap());

    let mut vertex_counter = 1u32;
    let mut vertices = HashMap::new();
    let mut objs = Vec::new();
    for line in file.lines() {
        let l = line.unwrap();
        if l.starts_with("v ") {
            let coords: Vec<f32> = l.split(' ').skip(1).filter_map(|s| str::parse(s).ok()).collect();
            assert!(coords.len() == 3);
            vertices.insert(vertex_counter, Point {
                x: coords[0],
                y: coords[1],
                z: coords[2]
            });
            vertex_counter += 1;
        } else if l.starts_with("f ") {
            let points: Vec<&Point> = l.split(' ').skip(1).map(|w| chop(w, '/')).filter_map(|s| str::parse::<u32>(s).ok()).filter_map(|p| vertices.get(&p)).collect();
            assert!(points.len() == 3);
            objs.push(SceneObject {
                shape: Shape::Triangle {
                    p1: points[0].clone(),
                    p2: points[1].clone(),
                    p3: points[2].clone()
                },
                properties: MaterialProperties {
                    color_primary: Color {
                        red: 0xFF,
                        green: 0x00,
                        blue: 0x00
                    },
                    color_secondary: Color::black(),
                    specular: 1.0,
                    diffuse: 0.8,
                    ambient: 0.2,
                    shininess: 13.0,
                    reflectivity: 0.5
                }
            });
        }
    }
    println!("File parsed."); // XXX
    objs
}

fn chop(s: &str, c: char) -> &str {
    s.splitn(1, c).next().unwrap()
}
