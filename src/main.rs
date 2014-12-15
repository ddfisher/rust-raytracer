// TODO: remove once stable
#![allow(dead_code)]

extern crate png;

use std::vec::Vec;
use std::num::{from_uint, Float, FloatMath};

//TODO: consider making multithreaded

#[deriving(Clone)]
struct Pixel {
    red: u8,
    green: u8,
    blue: u8
}

struct Image {
    width: uint,
    height: uint,
    pixels: Vec<Pixel>
}

impl Pixel {
    fn new(r: u8, g: u8, b: u8) -> Pixel {
        Pixel { red: r, green: g, blue: b }
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
            pixels: Vec::from_elem(width * height, Pixel::new(0, 0, 0))
        }
    }

    // TODO: try to remove some copying?
    fn to_png_image(&self) -> png::Image {
        png::Image {
            width: from_uint(self.width).expect("width number out of range for u32"),
            height: from_uint(self.height).expect("height number out of range for u32"),
            pixels: png::PixelsByColorType::RGB8(self.pixels.iter().flat_map(|ref p| p.bytes().into_iter()).collect())
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
    let width = 300;
    let height = 300;
    let mut img = Image::new(width, height);
    let scene = setup_scene();

    for x in range(0, width) {
        for y in range(0, height) {
            let ray = pixel_to_ray(x, y, width, height);
            if scene.hit(&ray).is_some() {
                img.pixels[x + y * width] = Pixel{red:0xFF, green: 0xFF, blue: 0xFF}
            }
        }
    }

    img
}

struct Point {
    x: f32,
    y: f32,
    z: f32
}

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

enum SceneObject {
    Sphere {
        center: Point,
        radius: f32
    }
}

impl SceneObject {
    fn hit(&self, &Ray{origin: ref o, direction: ref d}: &Ray) -> Option<Point> {
        // TODO: don't treat the ray like a line
        // TODO: optimize
        match self {
            &SceneObject::Sphere {center: ref c, radius: ref r} => {
                let offset = *o - *c;
                let initial = -d.dot(&offset);
                let under_radical = d.dot(&offset).powi(2) - offset.length().powi(2) + r.powi(2);
                if under_radical < 0.0 {
                    None
                } else if under_radical == 0.0 {
                    Some(*o + d.times(initial))
                } else {
                    Some(*o + d.times(FloatMath::min(initial + under_radical, initial - under_radical)))
                }
            }
        }
    }
}

struct Scene {
    objects: Vec<SceneObject>
}

impl Scene {
    fn hit(&self, ray: &Ray) -> Option<(Point, &SceneObject)> {
        self.objects.iter()
            .filter_map(|obj| obj.hit(ray).map(|p| (p,obj)))
            .min_by(|&(ref p,_)| OrderedF32(p.distance(&ray.origin)))
    }
}

// TODO: don't use this stupid projection
fn pixel_to_ray(x: uint, y: uint, width: uint, height: uint) -> Ray {
    Ray {
        origin: Point {
            x: x as f32 / width as f32 * 2.0 - 1.0,
            y: y as f32 / height as f32 * 2.0 - 1.0,
            z: -1.0
        },
        direction: Vector {
            dx: 0.0,
            dy: 0.0,
            dz: 1.0
        }
    }
}

fn setup_scene() -> Scene {
    Scene {
        objects: vec![
            SceneObject::Sphere{
                center: Point {x:0.0, y:0.0, z:0.0},
                radius: 0.1
            }
        ]
    }
}
