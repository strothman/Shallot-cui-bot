# Krea 2 — Confirmed Celebrity Recognition Reference

> **Last updated:** September 2026
> **Source:** Community stress tests on r/StableDiffusion using the `Krea 2 Turbo FP8` checkpoint.
> Test methodology: `"Name (profession) chest up on gray background, look at camera"` — Euler sampler, seed 42, 20 steps, no ControlNet, no negative prompt.

---

## How to Read This List

Celebrities are grouped by gender, then by category (Actors, Musicians, etc.). Each entry is graded:

| Tier | Meaning |
|------|---------|
| ⭐⭐⭐ | **Excellent** — Highly recognizable on name alone. Distinctive features lock in. |
| ⭐⭐ | **Good** — Recognizable in most seeds. May need a profession tag for clarity. |
| ⭐ | **Usable** — Works in some seeds, but may drift or need compositional reference. |

> [!TIP]
> Keep prompts short. `"Scarlett Johansson (actress)"` beats a 3-paragraph LLM expansion. Add **profession** when the name could be ambiguous.

> [!IMPORTANT]
> These results reflect the **base Krea 2 Turbo FP8 model's inherent training knowledge** — no LoRAs, no IP-Adapter, no ControlNet. Your mileage will improve significantly with compositional reference or identity edit tools.

---

## 👨 Male

### Actors — Classic / Legendary

| Name | Tier | Notes |
|------|------|-------|
| Marlon Brando | ⭐⭐⭐ | Iconic jawline and expression. One of the best. |
| Robert De Niro | ⭐⭐⭐ | Very strong across all ages. |
| Al Pacino | ⭐⭐⭐ | Distinct features render reliably. |
| Jack Nicholson | ⭐⭐⭐ | Eyebrows and grin are unmistakable. |
| Anthony Hopkins | ⭐⭐⭐ | Strong recognition. |
| Sean Connery | ⭐⭐⭐ | Classic Bond look locks in well. |
| Morgan Freeman | ⭐⭐⭐ | Extremely distinctive — freckles, voice implied through presence. |
| Kirk Douglas | ⭐⭐ | Chin dimple helps. |
| Laurence Olivier | ⭐⭐ | Best with era context (e.g. "1940s"). |
| Sidney Poitier | ⭐⭐ | Good recognition but can drift age. |
| Christopher Lee | ⭐⭐⭐ | Tall, angular — very recognizable. |
| Vincent Price | ⭐⭐⭐ | Mustache + arch expression = perfect hit. |
| Peter Cushing | ⭐⭐ | Good, especially with horror context. |
| Peter Sellers | ⭐⭐ | Works, but may pull different characters. |
| Paul Newman | ⭐⭐⭐ | Blue eyes, strong jaw — locks in. |

### Actors — Modern A-List

| Name | Tier | Notes |
|------|------|-------|
| Tom Hanks | ⭐⭐⭐ | Extremely reliable across ages. |
| Denzel Washington | ⭐⭐⭐ | Strong, distinctive face. |
| Leonardo DiCaprio | ⭐⭐⭐ | Works well, may default to younger appearance. |
| Brad Pitt | ⭐⭐⭐ | Very consistent. |
| Johnny Depp | ⭐⭐⭐ | Highly distinctive styling helps. |
| Will Smith | ⭐⭐⭐ | Very strong recognition. |
| Harrison Ford | ⭐⭐⭐ | Indiana Jones era or older both work. |
| Samuel L. Jackson | ⭐⭐⭐ | Unmistakable. |
| Keanu Reeves | ⭐⭐⭐ | Long hair or short — both hit. |
| Robert Downey Jr. | ⭐⭐⭐ | Tony Stark era especially strong. |
| Hugh Jackman | ⭐⭐⭐ | Very recognizable. |
| Ryan Reynolds | ⭐⭐ | Good, but can blend with other "handsome white actors." |
| Ryan Gosling | ⭐⭐ | Same — recognizable but less distinctive features. |
| Chris Hemsworth | ⭐⭐ | Thor look helps. Can blur with Evans/Pratt. |
| Chris Evans | ⭐⭐ | Good with context. Same-face risk with other Chrises. |
| Chris Pratt | ⭐ | Weakest of the Chrises. Benefits from "beard" or "muscular." |
| Tom Holland | ⭐⭐ | Youth and build are distinctive enough. |
| Jake Gyllenhaal | ⭐⭐ | Works well with facial hair. |
| Christian Bale | ⭐⭐⭐ | Very distinctive across his transformations. |
| Tom Hardy | ⭐⭐ | Build helps, face can drift. |
| Matt Damon | ⭐⭐ | Decent but not the most distinctive. |
| George Clooney | ⭐⭐⭐ | Salt-and-pepper look is iconic. |
| Michael Douglas | ⭐⭐ | Good, especially Wall Street era. |
| Idris Elba | ⭐⭐⭐ | Strong recognition. |
| Benedict Cumberbatch | ⭐⭐⭐ | Unique bone structure helps enormously. |
| Oscar Isaac | ⭐⭐ | Good, benefits from beard. |
| Pedro Pascal | ⭐⭐ | Growing recognition. Mustache helps. |
| Cillian Murphy | ⭐⭐⭐ | Cheekbones and blue eyes are distinctive. Peaky Blinders era strong. |
| Adam Driver | ⭐⭐⭐ | Highly unique face — excellent recognition. |
| Rami Malek | ⭐⭐⭐ | Eyes are unmistakable. |
| Timothée Chalamet | ⭐⭐ | Works but may age inconsistently. |
| Robert Pattinson | ⭐⭐ | Good with jaw/hair context. |

### Actors — Action / Genre

| Name | Tier | Notes |
|------|------|-------|
| Arnold Schwarzenegger | ⭐⭐⭐ | Iconic physique and face. |
| Sylvester Stallone | ⭐⭐⭐ | Very distinctive. |
| Bruce Willis | ⭐⭐⭐ | Bald head = instant lock. |
| Dwayne Johnson | ⭐⭐⭐ | The Rock is unmistakable. |
| Jason Statham | ⭐⭐⭐ | Bald + stubble = strong hit. |
| Vin Diesel | ⭐⭐⭐ | Very recognizable. |
| Liam Neeson | ⭐⭐ | Good, benefits from age context. |
| Daniel Craig | ⭐⭐ | Bond context helps. |
| Ralph Fiennes | ⭐⭐ | Good but can drift. |
| Gary Oldman | ⭐⭐ | Chameleon actor — may not lock one look. |
| Mads Mikkelsen | ⭐⭐⭐ | Striking Nordic features. Excellent. |
| Christoph Waltz | ⭐⭐ | Good with smile/expression context. |
| Willem Dafoe | ⭐⭐⭐ | Extremely distinctive face. Top-tier. |
| Javier Bardem | ⭐⭐ | Good, especially No Country era. |

### Actors — Comedy

| Name | Tier | Notes |
|------|------|-------|
| Jim Carrey | ⭐⭐⭐ | Elastic face = perfect recognition. |
| Robin Williams | ⭐⭐⭐ | Very strong — hairy arms, kind eyes. |
| Eddie Murphy | ⭐⭐⭐ | Excellent across eras. |
| Adam Sandler | ⭐⭐ | Casual look helps. |
| Jack Black | ⭐⭐⭐ | Build + expressions = distinctive. |
| Will Ferrell | ⭐⭐ | Curly hair helps. |
| Steve Carell | ⭐⭐ | Good, especially with glasses. |
| Rowan Atkinson | ⭐⭐⭐ | Mr. Bean face is iconic. |
| Sacha Baron Cohen | ⭐⭐ | Borat look helps, otherwise may drift. |
| Dave Chappelle | ⭐⭐ | Good recognition. |

### Musicians — Male

| Name | Tier | Notes |
|------|------|-------|
| Freddie Mercury | ⭐⭐⭐ | Mustache era = perfect. One of the absolute best. |
| David Bowie | ⭐⭐⭐ | Ziggy Stardust or Thin White Duke both excellent. |
| Prince | ⭐⭐⭐ | Iconic styling. |
| Michael Jackson | ⭐⭐⭐ | Very strong recognition. May default to Thriller era. |
| Elton John | ⭐⭐⭐ | Glasses + flamboyant style = instant hit. |
| Elvis Presley | ⭐⭐⭐ | Universally works — pompadour is iconic. |
| John Lennon | ⭐⭐⭐ | Round glasses era is iconic. |
| Paul McCartney | ⭐⭐ | Better with era context. |
| Mick Jagger | ⭐⭐⭐ | Lips are unmistakable. |
| Bob Dylan | ⭐⭐ | Young or old both work, but can drift. |
| Frank Sinatra | ⭐⭐⭐ | Fedora + blue eyes = strong. |
| Johnny Cash | ⭐⭐⭐ | Man in Black is iconic. |
| Bob Marley | ⭐⭐⭐ | Dreads = instant recognition. |
| Eminem | ⭐⭐⭐ | Blonde era especially strong. |
| Snoop Dogg | ⭐⭐⭐ | Very distinctive look. |
| Kanye West | ⭐⭐⭐ | Strong recognition. |
| Jay-Z | ⭐⭐ | Good but benefits from context. |
| Ed Sheeran | ⭐⭐ | Red hair helps. |
| Harry Styles | ⭐⭐ | Good, may age up. |
| Bruno Mars | ⭐⭐ | Good with hat/pompadour. |
| Bruce Springsteen | ⭐⭐ | Good, bandana/guitar context helps. |
| Keith Richards | ⭐⭐⭐ | Weathered face is iconic. |
| Willie Nelson | ⭐⭐⭐ | Braids + bandana = perfect. |
| Kendrick Lamar | ⭐⭐ | Good, cornrows help. |

---

## 👩 Female

### Actresses — Classic / Legendary

| Name | Tier | Notes |
|------|------|-------|
| Audrey Hepburn | ⭐⭐⭐ | Breakfast at Tiffany's look = perfect. One of the best. |
| Marilyn Monroe | ⭐⭐⭐ | Blonde curls + beauty mark = unmistakable. |
| Elizabeth Taylor | ⭐⭐⭐ | Violet eyes and dark hair — very strong. |
| Sophia Loren | ⭐⭐⭐ | Classic Italian beauty — excellent recognition. |
| Ingrid Bergman | ⭐⭐ | Good with era context. |
| Grace Kelly | ⭐⭐ | Good, may need "1950s" context. |
| Meryl Streep | ⭐⭐⭐ | Very strong across all ages. |
| Julia Roberts | ⭐⭐⭐ | Big smile = instant recognition. |

### Actresses — Modern A-List

| Name | Tier | Notes |
|------|------|-------|
| Scarlett Johansson | ⭐⭐⭐ | Very distinctive features. Excellent. |
| Angelina Jolie | ⭐⭐⭐ | Lips and cheekbones = unmistakable. |
| Nicole Kidman | ⭐⭐⭐ | Red hair era or blonde both strong. |
| Cate Blanchett | ⭐⭐⭐ | Regal features — very consistent. |
| Natalie Portman | ⭐⭐⭐ | Pixie or long hair — both work. |
| Anne Hathaway | ⭐⭐ | Good, big eyes help. |
| Margot Robbie | ⭐⭐ | Good, but can blend with other blonde actresses. |
| Charlize Theron | ⭐⭐⭐ | Strong bone structure = reliable. |
| Kate Winslet | ⭐⭐ | Good, especially Titanic era. |
| Sandra Bullock | ⭐⭐ | Decent but not the most distinctive. |
| Jennifer Lawrence | ⭐⭐ | Good, benefits from specific era context. |
| Emma Stone | ⭐⭐ | Red hair era is strongest. |
| Florence Pugh | ⭐⭐ | Growing in recognition. Distinctive features help. |
| Anya Taylor-Joy | ⭐⭐⭐ | Wide-set eyes are highly distinctive. |
| Zendaya | ⭐⭐⭐ | Very recognizable. Modern icon. |
| Viola Davis | ⭐⭐⭐ | Strong, powerful features — excellent. |
| Lupita Nyong'o | ⭐⭐⭐ | Stunning bone structure = consistent hits. |
| Angela Bassett | ⭐⭐⭐ | Instantly recognizable. |
| Halle Berry | ⭐⭐⭐ | Short hair era especially strong. |
| Tilda Swinton | ⭐⭐⭐ | Androgynous features are unmistakable. |
| Helena Bonham Carter | ⭐⭐⭐ | Wild hair and aesthetic = instant lock. |
| Jessica Chastain | ⭐⭐ | Red hair helps. Can blend with Bryce Dallas Howard. |
| Cameron Diaz | ⭐⭐ | 90s/2000s era is strongest. |
| Drew Barrymore | ⭐⭐ | Good, 90s era especially. |
| Reese Witherspoon | ⭐⭐ | Chin is distinctive. |
| Dakota Johnson | ⭐ | Can drift. Not highly distinctive. |
| Kristen Stewart | ⭐⭐ | Androgynous styling helps. |
| Saoirse Ronan | ⭐⭐ | Good with era/movie context. |
| Emma Watson | ⭐⭐ | Good, Hermione era especially strong. |
| Michelle Pfeiffer | ⭐⭐⭐ | Catwoman era = excellent. |

### Actresses — Action / Genre

| Name | Tier | Notes |
|------|------|-------|
| Sigourney Weaver | ⭐⭐⭐ | Alien Ripley = iconic. |
| Gal Gadot | ⭐⭐⭐ | Wonder Woman look is very strong. |
| Zoe Saldana | ⭐⭐ | Good, but can be generic without context. |
| Tessa Thompson | ⭐⭐ | Good with Valkyrie context. |

### Musicians — Female

| Name | Tier | Notes |
|------|------|-------|
| Beyoncé | ⭐⭐⭐ | Universally recognizable. Excellent. |
| Rihanna | ⭐⭐⭐ | Extremely strong across all eras. |
| Taylor Swift | ⭐⭐⭐ | Very consistent. Blonde or red lip era both work. |
| Lady Gaga | ⭐⭐⭐ | Dramatic styling = perfect recognition. |
| Adele | ⭐⭐⭐ | Winged eyeliner + presence = strong. |
| Billie Eilish | ⭐⭐⭐ | Green/black hair era is iconic. |
| Ariana Grande | ⭐⭐⭐ | Ponytail is a strong visual cue. |
| Dua Lipa | ⭐⭐ | Good, dark features help. |
| Katy Perry | ⭐⭐ | Good, especially dark hair era. |
| Dolly Parton | ⭐⭐⭐ | Unmistakable. Big hair + rhinestones. |
| Tina Turner | ⭐⭐⭐ | Wild hair + powerful stance = iconic. |
| Aretha Franklin | ⭐⭐ | Good with era context. |
| Diana Ross | ⭐⭐⭐ | 70s disco era = excellent. |
| Nina Simone | ⭐⭐ | Good with context. |
| Joni Mitchell | ⭐⭐ | Blonde/70s era works best. |
| Nicki Minaj | ⭐⭐⭐ | Very distinctive styling. |
| Cardi B | ⭐⭐ | Good, dramatic looks help. |
| Jennifer Lopez | ⭐⭐⭐ | Very strong recognition. |
| Selena Gomez | ⭐⭐ | Good, but can be generic. |
| Miley Cyrus | ⭐⭐ | Short hair era is more distinctive. |
| Lana Del Rey | ⭐⭐ | Vintage aesthetic helps. |
| SZA | ⭐⭐ | Growing recognition. |
| Shania Twain | ⭐⭐ | 90s era especially strong. |
| Carrie Underwood | ⭐ | Can drift. Benefits from country context. |
| Salma Hayek | ⭐⭐⭐ | Very distinctive features. Excellent. |
| Penélope Cruz | ⭐⭐⭐ | Strong Mediterranean features = reliable. |
| Charli XCX | ⭐⭐ | Brat era helps with modern recognition. |
| Rosalía | ⭐⭐ | Good with flamenco/dramatic context. |

---

## 🎯 Quick-Reference: Top 10 Most Reliable (Name-Only, No LoRA)

### Male
1. Freddie Mercury
2. David Bowie
3. Willem Dafoe
4. Arnold Schwarzenegger
5. Morgan Freeman
6. Samuel L. Jackson
7. Jim Carrey
8. Dwayne Johnson
9. Keanu Reeves
10. Johnny Depp

### Female
1. Angelina Jolie
2. Scarlett Johansson
3. Beyoncé
4. Lady Gaga
5. Tilda Swinton
6. Dolly Parton
7. Rihanna
8. Halle Berry
9. Anya Taylor-Joy
10. Helena Bonham Carter

---

## ⚠️ Known Pitfalls

| Issue | Details |
|-------|---------|
| **Same-Face Syndrome** | Modern actors with generic "handsome/pretty" features (Chris Pratt, Margot Robbie, Dakota Johnson) can blur together or produce a generic face. |
| **Age Drift** | The model may default to the most famous era of a celebrity. Leo DiCaprio often renders young; Morgan Freeman often renders older. |
| **Common Names** | Names like "Chris Evans" or "Emma Watson" may occasionally pull non-celebrity references. Adding "(actor)" or "(actress)" fixes this. |
| **Deceased Subjects** | Historical figures (Marilyn Monroe, James Dean) tend to work well since their iconic images are frozen in time. |
| **Less Globally Famous** | B-list or regional celebrities may not have enough training representation. LoRAs are the solution. |

---

## 💡 Prompting Tips for Celebrity Generation

1. **Keep it short**: `"Scarlett Johansson (actress) sitting at a cafe, natural lighting"` > a 200-word description.
2. **Add profession**: Prevents model confusion, especially for common names.
3. **Specify era** when it matters: `"David Bowie (singer, Ziggy Stardust era)"`.
4. **Use compositional reference** for stronger identity locking (IP-Adapter, `/blend-krea`).
5. **For private characters** (not celebrities), always use a trained LoRA — name-only prompting won't work.

---

> [!NOTE]
> This list is curated from community testing results and does not represent an endorsement of generating non-consensual imagery of real people. Always respect rights of publicity and platform policies.
