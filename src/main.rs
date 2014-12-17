// TODO: remove once stable
#![allow(dead_code)]

extern crate png;

use std::vec::Vec;
use std::num::{from_uint, Float, Int, FloatMath};

//TODO: consider making multithreaded

// TODO: consider using a more fine grained Color for computation
#[deriving(Show, Clone)]
struct Color {
    red: u8,
    green: u8,
    blue: u8
}

impl Add<Color, Color> for Color {
    fn add(&self, other: &Color) -> Color {
        Color {
            red:   self.red.saturating_add(other.red),
            green: self.green.saturating_add(other.green),
            blue:  self.blue.saturating_add(other.blue)
        }
    }
}

impl Sub<Color, Color> for Color {
    fn sub(&self, other: &Color) -> Color {
        Color {
            red:   self.red.saturating_sub(other.red),
            green: self.green.saturating_sub(other.green),
            blue:  self.blue.saturating_sub(other.blue)
        }
    }
}

impl Mul<f32, Color> for Color {
    fn mul(&self, &f: &f32) -> Color {
        fn mul_sat(n: u8, f: f32) -> u8 {
            let p = n as f32 * f;
            let max: u8 = Int::max_value();
            if p >= (max + 1) as f32 { p as u8 } else { 0xFF }
        }

        Color {
            red:   mul_sat(self.red,   f),
            green: mul_sat(self.green, f),
            blue:  mul_sat(self.blue,  f)
        }
    }
}

struct Image {
    width: uint,
    height: uint,
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
    fn new(width: uint, height: uint) -> Image {
        Image {
            width: width,
            height: height,
            pixels: Vec::from_elem(width * height, Color::new(0, 0, 0))
        }
    }

    // TODO: try to remove some copying?
    fn to_png_image(&self) -> png::Image {
        let mut pixel_vector: Vec<u8> = Vec::with_capacity(self.width * self.height * 3);
        for pixel in self.pixels.iter() {
            pixel_vector.push(pixel.red);
            pixel_vector.push(pixel.green);
            pixel_vector.push(pixel.blue);
        }
        png::Image {
            width: from_uint(self.width).expect("width number out of range for u32"),
            height: from_uint(self.height).expect("height number out of range for u32"),
            pixels: png::PixelsByColorType::RGB8(pixel_vector)
        }
    }
}

fn main() {
    let path = Path::new("scene.png");
    let image = raytrace();
    let mut png = image.to_png_image();

    png::store_png(&mut png, &path).unwrap();
}

fn raytrace() -> Image {
    let width = 900;
    let height = 900;
    let mut img = Image::new(width, height);
    let scene = setup_scene();

    for x in range(0, width) {
        for y in range(0, height) {
            let ray = pixel_to_ray(x, y, width, height);
            let pixel = ray_to_color(&ray, &scene);
            img.pixels[x + y * width] = pixel;
        }
    }

    img
}

#[deriving(Show, Clone)]
struct Point {
    x: f32,
    y: f32,
    z: f32
}

#[deriving(Show, Clone)]
struct Vector {
    dx: f32,
    dy: f32,
    dz: f32
}

impl Point {
    fn distance(&self, other: &Point) -> f32 {
        (*self - *other).length()
    }
}

impl Vector {
    fn length(&self) -> f32 {
        Float::sqrt(self.dx * self.dx + self.dy * self.dy + self.dz * self.dz)
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

    fn times(&self, scalar: f32) -> Vector {
        Vector {
            dx: self.dx * scalar,
            dy: self.dy * scalar,
            dz: self.dz * scalar
        }
    }
}

impl Add<Vector, Vector> for Vector {
    fn add(&self, other: &Vector) -> Vector {
        Vector {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            dz: self.dz + other.dz
        }
    }
}

impl Sub<Vector, Vector> for Vector {
    fn sub(&self, other: &Vector) -> Vector {
        Vector {
            dx: self.dx - other.dx,
            dy: self.dy - other.dy,
            dz: self.dz - other.dz
        }
    }
}

impl Add<Vector, Point> for Point {
    fn add(&self, rel: &Vector) -> Point {
        Point {
            x: self.x + rel.dx,
            y: self.y + rel.dy,
            z: self.z + rel.dz
        }
    }
}

impl Sub<Point, Vector> for Point {
    fn sub(&self, other: &Point) -> Vector {
        Vector {
            dx: self.x - other.x,
            dy: self.y - other.y,
            dz: self.z - other.z
        }
    }
}


#[deriving(PartialEq, PartialOrd)]
struct OrderedF32(f32);

impl Eq for OrderedF32{} // because I'm a bad person

impl Ord for OrderedF32 {
    fn cmp(&self, &OrderedF32(other): &OrderedF32) -> Ordering {
        let &OrderedF32(s) = self;
        match (s.is_nan(), other.is_nan()) {
            (true, true)   => Equal,
            (true, false)  => Less,
            (false, true)  => Greater,
            (false, false) => if s == other { Equal } else if s < other { Less } else { Greater }
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
    }
}

struct MaterialProperties {
    color: Color,
    specular: f32,
    diffuse: f32,
    ambient: f32,
    shininess: f32
}

// TODO: consider other values of epsilon
static EPSILON: f32 = 0.0000001;
static SELF_INTERSECT_OFFSET: f32 = 0.00001;

impl SceneObject {
    fn hit(&self, &Ray{origin: ref o, direction: ref d}: &Ray) -> Option<Point> {
        // TODO: don't treat the ray like a line
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
            }
        }
    }

    fn normal_at(&self, point: &Point) -> Vector {
        match self.shape {
            Shape::Sphere {ref center, ..} => {
                (*point - *center).normalized()
            },
            Shape::Plane {ref normal, ..} => normal.clone()
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
            .min_by(|&(ref p,_)| OrderedF32(p.distance(&ray.origin)))
    }

    fn hit_any(&self, ray: &Ray) -> bool {
        self.objects.iter()
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
    fn vector_for(&self, point: &Point) -> Vector {
        match self {
            &Light::Direction {ref direction, ..} => direction.normalized().times(-1.0)
        }
    }

    fn intensity_for(&self, point: &Point) -> f32 {
        match self {
            &Light::Direction {intensity, ..} => intensity
        }
    }
}

// TODO: fix up the projection
fn pixel_to_ray(x: uint, y: uint, width: uint, height: uint) -> Ray {
    let camera = Point {
        x: 0.0,
        y: 0.0,
        z: -4.0
    };
    Ray {
        direction: (Point {
            x: x as f32 / width as f32 * 2.0 - 1.0,
            y: (height - y) as f32 / height as f32 * 2.0 - 1.0,
            z: -3.0
        } - camera).normalized(),
        origin: camera
    }
}

fn setup_scene() -> Scene {
    Scene {
        objects: vec![
            SceneObject {
                shape: Shape::Sphere {
                    center: Point {x:0.0, y:0.0, z:0.0},
                    radius: 1.0
                },
                properties: MaterialProperties {
                    color: Color {
                        red: 0xFF,
                        green: 0x00,
                        blue: 0x00
                    },
                    specular: 1.0,
                    diffuse: 0.8,
                    ambient: 0.2,
                    shininess: 3.0
                }
            },
            SceneObject {
                shape: Shape::Plane {
                    point: Point {x:0.0, y:-2.0, z:0.0},
                    normal: Vector {dx: 0.0, dy: 1.0, dz:0.0}
                },
                properties: MaterialProperties {
                    color: Color {
                        red: 0x50,
                        green: 0x30,
                        blue: 0xA0
                    },
                    specular: 0.0,
                    diffuse: 1.0,
                    ambient: 0.2,
                    shininess: 2.0
                }
            }
        ],
        lights: vec![
            // Light::Direction {
            //     direction: Vector {
            //         dx: -2.0,
            //         dy: -1.0,
            //         dz: 0.0
            //     },
            //     intensity: 0.2
            // },
            // Light::Direction {
            //     direction: Vector {
            //         dx: 1.5,
            //         dy: -3.0,
            //         dz: 1.0
            //     },
            //     intensity: 0.8
            // }
            Light::Direction {
                direction: Vector {
                    dx: 0.0,
                    dy: 0.0,
                    dz: 1.0
                },
                intensity: 0.8
            }
        ]
    }
}

fn ray_to_color(ray: &Ray, scene: &Scene) -> Color {
    fn light_contribution(light: &Light, point: &Point, obj: &SceneObject, ray: &Ray) -> Color {
        // http://en.wikipedia.org/wiki/Lambertian_reflectance
        let norm_v = obj.normal_at(point);
        let light_v = &light.vector_for(point);
        let diffuse_intensity = obj.properties.diffuse * FloatMath::max(0.0, norm_v.dot(light_v));
        // http://en.wikipedia.org/wiki/Phong_reflection_model
        let light_reflection_v = norm_v.times(2.0 * norm_v.dot(light_v)) - *light_v;
        let specular_intensity = obj.properties.specular
            * FloatMath::max(0.0, -light_reflection_v.dot(&ray.direction.normalized())).powf(obj.properties.shininess);
        let diffuse_color = obj.properties.color * diffuse_intensity;
        let specular_color = (Color::white() - diffuse_color) * specular_intensity;
        (diffuse_color + specular_color) * light.intensity_for(point)
    }

    // TODO: remove duplicate normal calls
    match scene.hit(ray) {
        Some((point, obj)) => {
            let ambient = obj.properties.color * obj.properties.ambient;
            ambient + scene.lights.iter()
                .filter(|l| {
                    let purturbed_point = point + obj.normal_at(&point).times(SELF_INTERSECT_OFFSET);
                    !scene.hit_any(&Ray {direction: l.vector_for(&purturbed_point), origin: purturbed_point})
                })
                .map(|l| light_contribution(l, &point, obj, ray))
                .fold(Color::black(), |a, b| a + b)
        },
        None => Color::black()
    }
}
