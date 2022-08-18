/// Algorithm to make a heat map
/// For a pixelated rectangle
/// Heat sources at some pixels Each pixel can have 0 or one heat
/// sources
/// The temperature at pixle p because of heat at pixel h is:
/// heat(h)/(1+distance(p, h))
/// Define queue of rectangles, Qi, with the main rectangle in it.
/// Qr and Qo are empty queues of rectangles
/// While Qi is not empty
/// remove rectangle R from Qi
/// If R is <= minimum size or R contains no heat sources put R in Qr
/// Else quater R into four equal sized rectangles and put them all in Qi
/// Discard R
/// End While
/// While Qr is not empty
/// Remove rectangle R from Qr
/// If R <= minimum size calculate heat(R) and put R on Qo and next loop
/// Calculate heat at four corners of R: h1 h2 h3 h4
/// If h1 = h2 = h3 = h4  heat(R) = h1 and put on Qo and next loop
/// Quater R into four equal rectangles and put them in Qr, 
/// Discard R
/// End While
/// Paint Qo on screen

///  Some things to help with optimisation
/// [Vec::get_unchecked](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.get_unchecked)
/// [Vec::as_mut_ptr](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.as_mut_ptr)

extern crate image;

use std::time::{Instant};
#[allow(unused_imports)]
use conv::*;
use image::{Rgb, RgbImage};
use rand::{Rng};
#[allow(unused_imports)]
use imageproc::drawing::draw_line_segment_mut;
#[allow(unused_imports)]
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::drawing::draw_filled_rect_mut;
#[allow(unused_imports)]
use imageproc::drawing::draw_filled_circle_mut;
use imageproc::rect;
use std::collections::HashMap;
use std::convert::TryInto;
use std::env;
use std::path::Path;
use std::primitive::u8;
use std::fmt;
//use rand::distributions::{Alphanumeric, Uniform, Standard};
const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;
const MAX_HEAT: usize = 100_000;
const MINIMUM_SIZE: usize = 1;
const MAXIMUM_SIZE:usize = 1000;

/// A location
#[derive(PartialEq, Hash, Eq, Clone, Copy, Debug)]
struct Pixel {
    x: usize,
    y: usize,
}
impl fmt::Display for Pixel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}x{})", self.x, self.y)
    }
}

/// Store temperature at pixels.  Only some locations get calculated.
/// TODO: Move colour handling code from here
struct TemperatureCache {
    data:HashMap<Pixel, usize>,
    palette:HashMap<usize, Rgb<u8>>,
}
impl TemperatureCache {
    fn get_colour_index(&self, heat:usize)-> usize{
	let mut res: usize = 0;
	let mut keys:Vec<usize> = self.palette.iter().
	    map(|k| *k.0).collect::<Vec<usize>>();
	keys.sort();
	
	for k in keys.iter().rev() {
	    if heat > *k {
		res = *k;
		break;
	    }
	}
	// `res` is set to the largest key in palette that is smaller
	// than `heat`.  If there are none it is already 0, the
	// smallest (bluest) band
	res	
    }

    fn set(&mut self, at:&Pixel, colour_index:usize){
	self.data.insert(*at, colour_index);
    }
    
    #[allow(dead_code)]
    fn get(&self, at:&Pixel) ->	Option<&usize> {
	self.data.get(at)
    }
    
    // TemperatureCache
    fn calculate_temp(at:&Pixel, heat_sources:&Vec<HeatSource>) -> usize {
	let temp = heat_sources.iter().
	    fold(0, |a, b| a + b.temperature_from(at.x, at.y));
	temp
    }

    // Get the colour to paint a Pixel.  If it does not exist,
    // calcultate it, cache it and return it.  The value returned is
    // the key fo the palette
    fn get_set(&mut self, 
	       at:&Pixel, heat_sources:&Vec<HeatSource>) -> usize {
	match
	    self.data.get(at) {
		Some(h) => *h,
		None => {
		    let h = TemperatureCache::calculate_temp(at, heat_sources);
		    let res: usize = self.get_colour_index(h);
		    self.set(at, res);
		    h
		},
	    }
    }
    
    #[allow(dead_code)]
    fn	add(& mut self, at: &Pixel, heat_sources: &Vec<HeatSource>) {
	self.set(at, TemperatureCache::calculate_temp(at, heat_sources));
    }

    fn new(heat_sources:&Vec<HeatSource>)  -> TemperatureCache{
	let mut samples:Vec<(Pixel, usize)> = Vec::new();
	// Sample data to make colour palette
	for _ in 0..100 {
	    let x = rand::thread_rng().gen_range(0..WIDTH);
	    let y = rand::thread_rng().gen_range(0..HEIGHT);
	    let pixel = Pixel{x, y};
	    let h = TemperatureCache::calculate_temp(&pixel, heat_sources);
	    samples.push((pixel, h));
	}
	samples.sort_by(|a, b|  a.1.partial_cmp(&b.1).unwrap());

	// Want X colours. So divide have X - 1 heat values as well as
	// zero, associate each with a colour: blue => 0 and red => hottest.
	let colour_count = 8;
	let div = samples.len() / (colour_count - 1);
	let mut palette: HashMap<usize, Rgb<u8>> = HashMap::new();	

	// The bluest value is indexed by zero
	palette.insert(0, Rgb::from([0, 0, u8::MAX]));

	for i in 0..(colour_count) {
	    let key = (div * i) as usize;

	    let value = Rgb::<u8>::from(
		[
		    // Red. 
		    (i * (u8::MAX as usize / (colour_count - 1))) as u8,

		    // Green
		    if i < colour_count/2 {
			(i * 2 * (u8::MAX as usize / (colour_count - 1))) as u8
		    }else{
			(
			    ((colour_count - 1) - 2*(i - colour_count/2) as usize)  * u8::MAX as usize /
				(colour_count -1)
			) as u8

		    },

		    // Blue
		    (
			((colour_count - 1) - i as usize)  * u8::MAX as usize /
			    (colour_count -1)
		    ) as u8
		]
	    );
	    palette.insert(samples[key].1, value);
	}
	// (d - 1) * u8::MAX
	// div * u8::MAX
	let data = HashMap::new();
	
	TemperatureCache { data, palette}
    }
}

#[derive(Debug)]
struct Rect{

    // Optimisation: Make these references.  Would it make a    difference?
    loc:[Pixel;4], // [TL, TR, BR, BL] Clockwise from top left
    // width:usize,
    // height:usize,
    colour: Option<Rgb<u8>>,
}
impl Rect {
    fn width(&self) -> usize { self.loc[1].x - self.loc[0].x }

    // Rect
    fn height(&self) -> usize {
	// Counting starts at top left
	 self.loc[2].y - self.loc[1].y }

    fn area(&self) -> usize { self.height() * self.width() }

    // Rect
    #[allow(dead_code)]
    fn smaller_than(&self, other:&Rect) -> bool {
	if self.area() < other.area() { true }else{ false }
    }

    // Rect
    /// Test if a heat source at `loc` is influencing a rectangle.  
    fn contains(&self, loc:Pixel) -> bool {
	
	let result =
	    loc.x >= self.loc[0].x &&
	    loc.x <= self.loc[1].x &&
	    loc.y >= self.loc[0].y &&
	    loc.y <= self.loc[3].y;
	result
    }

    // Rect
    fn new_wh(x:usize, y:usize, width:usize,
    height:usize, colour:Option<Rgb<u8>>) -> Rect {

	Rect{loc:[
	    Pixel{x:x, y:y}, 
	    Pixel{x:x + width, y:y}, 
	    Pixel{x:x + width, y:y + height}, 
	    Pixel{x:x, y:y + height},
	],
	     colour,
	}
    }

    // Rect
    fn quarter(&self) -> [Rect ; 4] { //(Rect,Rect,Rect,Rect,){
	// Result in four rectangles
	// r1 r2
	// r3 r4	

	let xl = self.loc[0].x;
	let yt = self.loc[0].y;

	// If width or height of self is odd then the resulting
	// rectangle quarters will not have same width/height
	let w1:usize;
	let w2:usize;
	let h1:usize;
	let h2:usize;
	w1 = self.width()/2;
	if self.width() % 2 == 1 {
	    // Odd width
	    w2 = w1 + 1;
	}else{
	    w2 = w1;
	}

	h1 = self.height() / 2;
	if self.height() % 2 == 1 {
	    // Odd height
	    h2 = h1 + 1;
	}else{
	    h2 = h1;
	}

	// r1 r2
	// r3 r4	

	// r1
	[
	    Rect::new_wh(xl, yt, w1, h1, None),

	    // r2
	    Rect::new_wh(xl + w1, yt, w2, h1, None),

	    // r3
	    Rect::new_wh(xl, yt + h1, w1, h2, None),

	    // r4
	    Rect::new_wh(xl+ w1, yt + h1, w2, h2, None),
	]
    }

    fn top_left(&self) -> Pixel {
	self.loc[0]
    }
    fn bottom_right(&self) -> Pixel {
	self.loc[2]
    }
    #[allow(dead_code)]
    fn bottom_left(&self) -> Pixel {
	self.loc[3]
    }

    // Rect
    fn set_colour(&mut self,  heat_cache:&TemperatureCache, heat_sources:&Vec<HeatSource>) {
	
	let temperature = heat_sources.iter().fold(0, |accum, hs| accum + self.heat_from_heat_source(hs));
	
	self.colour = match heat_cache.palette.get(
	    &heat_cache.get_colour_index(temperature)
	) {
	    Some(c) => {
		Some(*c)
	    },
	    None => {
		panic!("Cound not find colour for {}", temperature)
	    },
	};
    }

    // Rect
    /// Calculate the distance from the center of the rectangle to the
    /// Pixel.  This is used to calculate the heat that the rectangle
    /// receives from a heat source at that pixel
    fn mean_square_distance(&self, pixel:&Pixel) -> usize {
	// The pixels are on a grid and the distance is to the centre
	// of rectangle which is not on grid, for even sizes.  So
	// double al the numbers in the calculation and dicde by 4 at
	// the end
	let x_r = (2 * self.loc[0].x + self.width()) as isize;
	let y_r = (2 * self.loc[0].y + self.height()) as isize;
	let dx  = x_r - 2 * pixel.x as isize;
	let dy  = y_r - 2 * pixel.y as isize;
	let result  = (dx * dx + dy * dy) as usize/4;
	result
    }

    // Rect
    /// Calculate the heat from a HeatSource
    fn heat_from_heat_source(&self, heat_source: &HeatSource) -> usize{
	heat_source.heat/(self.mean_square_distance(&heat_source.loc) + 1)
    }
}
impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rect({}/{})",
	       self.top_left(), self.bottom_right(),
	       )
    }
}

#[allow(dead_code)]
fn display_quarters(quarters: &[Rect ; 4]) -> String {
    format!("{} {} {} {}", quarters[0], quarters[1], quarters[2], quarters[3])
}

struct HeatSource {
    loc:Pixel,
    heat:usize,
}
impl HeatSource {
    fn distance_sq(&self, x:usize, y:usize) -> usize {	
	(
	    (self.loc.x as isize - x as isize) *
		(self.loc.x as isize - x as isize) +
		(self.loc.y as isize - y as isize) *
		(self.loc.y as isize - y as isize)
	).abs().try_into().unwrap()
    }
    fn temperature_from(&self, x:usize, y:usize) -> usize {

	let denominator = 1 + self.distance_sq(x, y);
	let numerator = self.heat;
	if denominator == 0 {
	    usize::MAX
	}else{
	    numerator / denominator
	}
    }
    #[allow(dead_code)]
    fn generate_heat_sources(count:usize) -> Vec<HeatSource> {
	let mut result = Vec::new();
	for _ in 0..count {
	    let heat = rand::thread_rng().gen_range(1..=MAX_HEAT);
	    let x = rand::thread_rng().gen_range(0..WIDTH);
	    let y = rand::thread_rng().gen_range(0..HEIGHT);
	    result.push(HeatSource::new(Pixel{x, y}, heat));
	}
	result
    }
    // HeatSource
    // fn heat_to_colour(heat:usize) -> Rgb<u8> {
    // 	let colour_linear:f64 = if heat < MAX_HEAT {
    // 	    heat as f64 
    // 	}else{
    // 	    MAX_HEAT as f64
    // 	} / MAX_HEAT as f64;
	
    // 	let r:u8  = (u8::MAX as f64  * colour_linear).approx().unwrap();
    // 	let g:u8  = 0;
    // 	let b:u8  = (u8::MAX as f64 - u8::MAX as f64  * colour_linear).approx().unwrap();

    // 	let res = Rgb::from([r, g, b]);
    // 	res
	    
    // }
    fn new(loc:Pixel, heat:usize) -> HeatSource {

	HeatSource {
	    loc,
	    heat,
	}
    }
}
impl fmt::Display for HeatSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
	
        write!(f, "HeatSource({}:{})", self.loc, self.heat)
    }
}
/// Calculate  heat for every point and output image
// fn the_hard_way(heat_sources: &Vec<HeatSource>, path: &str, width:usize, height:usize) {
//     let fp:String = format!("{}_complete.png", &path);
//     let new_path = Path::new(fp.as_str());
//     let mut image = RgbImage::new(width.try_into().unwrap(),
// 				  height.try_into().unwrap());
//     for x in 0..width {
// 	for y in 0..height {
// 	    let mut heat:usize = 0;
// 	    for hs in heat_sources.iter() {
// 		heat += hs.temperature_from(x, y);
// 	    }
// 	    draw_filled_circle_mut(
// 		&mut image,
// 		(x.try_into().unwrap(), y.try_into().unwrap()), 1,
// 		HeatSource::heat_to_colour(heat));
// 	}
//     }
//     match image.save(new_path) {
// 	Ok(()) => (),
// 	Err(err) => eprintln!("Failed to write file: {:?}", err),
//     };
    
	
// }
fn draw_image(q_o: &Vec<Rect>, path: &Path, palette:&HashMap<usize, Rgb<u8>>, heat_sources: &Vec<HeatSource>){
    let mut image = RgbImage::new(WIDTH.try_into().unwrap(),
				  HEIGHT.try_into().unwrap());
    
    for r in q_o {
	let colour = r.colour.unwrap();
	
	if r.height() > 0 && r.width() > 0 {
	    draw_filled_rect_mut(
		&mut image,
		rect::Rect::at(r.top_left().x as i32,
			       r.top_left().y as i32
		).of_size(r.width() as u32, r.height() as u32),
		colour);
	    // draw_hollow_rect_mut(
	    // 	&mut image,
	    // 	rect::Rect::at(r.top_left.x as i32,
	    // 		       r.top_left.y as i32
	    // 	).of_size(r.width() as u32, r.height() as u32),
	    // 	Rgb::<u8>::from([0, 0, 0],));
	}
    }

    for hs in heat_sources.iter(){
	draw_line_segment_mut(
	    &mut image,
		(hs.loc.x as f32 - 1.5 ,
		 hs.loc.y as f32 - 1.5),
		(hs.loc.x as f32 + 1.5 ,
		 hs.loc.y as f32 + 1.5),
	    
	    Rgb::<u8>::from([0, 0, 0],)
	);	    
	draw_line_segment_mut(
	    &mut image,
		(hs.loc.x as f32 + 1.5 ,
		 hs.loc.y as f32 - 1.5),
		(hs.loc.x as f32 - 1.5 ,
		 hs.loc.y as f32 + 1.5),
	    
	    Rgb::<u8>::from([0, 0, 0],)
	);
    }

    
    // // Colour palette

    let pw = WIDTH/24;
    let px = WIDTH-pw;
    let mut py = HEIGHT/10;
    let mut keys:Vec<usize> = palette.iter().
	map(|k| *k.0).collect::<Vec<usize>>();
    let ph = HEIGHT/(2*keys.len());
    if ph > 0 && pw > 0{
	keys.sort();
	for k in keys.iter().rev() {
	    let colour = palette.get(k).unwrap();

	    draw_filled_rect_mut(
		&mut image,
		rect::Rect::at(px as i32,
			       py as i32
		).of_size(pw as u32, ph as u32),
		*colour,
	    );
	    py += ph;
	}	
    }else{
	eprintln!("No room for a palette");
    }
    match image.save(path) {
	Ok(()) => (),
	Err(err) => eprintln!("Failed to write file: {:?}", err),
    };
}    

fn main() {
    // Even width and height
    assert!(WIDTH%2 == 0);
    assert!(HEIGHT%2 == 0);
    
    let arg = if env::args().count() == 2 {
        env::args().nth(1).unwrap()
    } else {
        panic!("Please enter a target file path")
    };

    let path = &arg;
    let minimum_size = MINIMUM_SIZE;
    let maximum_size = MAXIMUM_SIZE;
    // Set the bounds 
    let area = Rect::new_wh(0, 0, WIDTH, HEIGHT, None);

    let mut q_i:Vec<Rect> = vec![area];
    let mut q_r:Vec<Rect> = vec![];
    let mut q_o:Vec<Rect> = vec![];
    

    // Some heat sources
    let heat_sources = HeatSource::generate_heat_sources(WIDTH/2);
    // let heat_sources = vec![
    // 	HeatSource::new(
    // 	    Pixel{x:WIDTH/2, y:HEIGHT/2,},
    // 	    MAX_HEAT,
    // 	),
    // 	// HeatSource::new(
    // 	//     Pixel{x:WIDTH/8+WIDTH/2, y:HEIGHT/2,},
    // 	//     MAX_HEAT,
    // 	// ),
    // 	// HeatSource::new(
    // 	//     Pixel{x:WIDTH/6+WIDTH/3, y:5*HEIGHT/11,},
    // 	//     MAX_HEAT,
    // 	// ),
    // ];
    // for hs in heat_sources.iter() {
    // 	println!("{}", hs);
    // }
    let mut temperature_cache = TemperatureCache::new(&heat_sources);
    
    println!("Starting timer");
    let now = Instant::now();

    while !q_i.is_empty() {
	let r = q_i.pop().unwrap();
	if r .height() == 0 || r.width() == 0{
	    continue;
	}
	if r.area() <= minimum_size {
	    q_r.push(r);
	}else if 
	    !heat_sources.iter().any(|h| r.contains(Pixel{
		x:h.loc.x, y:h.loc.y
	    })) {
		q_r.push(r);
	    }else{
		let quarters = r.quarter();
		q_i.append(&mut Vec::from(quarters));
	    }
    }
    // draw_image(&q_r, &path);
    // return;
    while !q_r.is_empty(){
	let mut r = q_r.pop().unwrap();
	if r .height() == 0 || r.width() == 0{
	    continue;
	}
	if r.area() <= minimum_size {
	    r.set_colour(&temperature_cache, &heat_sources, );
	    q_o.push(r);
	    // draw_image(&q_o, &path);
	}else {
	    let h1 = temperature_cache.get_set(&r.loc[0], &heat_sources);
	    let h2 = temperature_cache.get_set(&r.loc[1], &heat_sources);
	    let h3 = temperature_cache.get_set(&r.loc[2], &heat_sources);
	    let h4 = temperature_cache.get_set(&r.loc[3], &heat_sources);
	    if  r.area() > maximum_size ||
		h1 != h2 ||
		h1 != h3 ||
		h1 != h4
	    {
		let quarters:[Rect; 4] = r.quarter();
		q_r.append(&mut Vec::from(quarters));
	    }else{
		// All four verticies of the rectangle have the same heat
		r.set_colour(&temperature_cache,  &heat_sources, );
		q_o.push(r);
	    }
	}
    }

    

    let elapsed_time = now.elapsed();
    println!("Running slow_function() took {} micro seconds.", elapsed_time.as_micros());

    let fp = format!("{}.png", path);
    let new_path = Path::new(fp.as_str());
    draw_image(&q_o, &new_path, &temperature_cache.palette, &heat_sources);

    // println!("Hard way");
    // the_hard_way(&heat_sources, path, WIDTH, HEIGHT);
}
