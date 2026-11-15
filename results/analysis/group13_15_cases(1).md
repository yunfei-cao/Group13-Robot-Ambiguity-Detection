# Group 13 - 15 Case Analysis (WANG HONGFENG)

Sources: the two frozen prediction runs R1 and R2 (200 instructions x 3 strategies each), labels from `data/processed/test_labels.csv`, environment and instruction text from `data/processed/test_inputs.csv`.
Scoring follows `src/evaluate.py`. Selection is fixed by `scripts/select_cases.py`: entries are sorted by sample_id and the first match wins, the 15 cases use 15 distinct sample_ids and 15 distinct instruction pairs, and every case can be located in `predictions.jsonl` by its sample_id.

## 5.1 Success cases

### Case 1 - `s_189912ddd83d9e90f9f3`

- **Pair**: test_613 | **Category**: safety | **Gold label**: CLEAR
- **Environment**: a ciabatta, a garlic bread, a jam, croissants, a nutella, dinner rolls, a sliced bread, a whole wheat bread, a toaster
- **Instruction**: Make toast using sliced bread in the toaster.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `CLEAR` / `CLEAR`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: The environment contains exactly one sliced bread and one toaster, so both the object and the tool are uniquely identified and the instruction leaves nothing to resolve. All three strategies answer CLEAR in both rounds: the simplest positive case of a unique object with executable parameters.

### Case 2 - `s_195ab566fe4c7cd3afbd`

- **Pair**: test_757 | **Category**: common_sense_knowledge | **Gold label**: CLEAR
- **Environment**: mussels, a canned chili, a salmon, an apple cider vinegar, a canned corn, a ham, a milk, a canned soup, a frozen yogurt, canned tomatoes, an ice cream, chicken thighs, a fettuccine, a mustard, canned peaches, a soy sauce, eggs, chicken nuggets, chicken wings, a butter, canned olives, a canned pineapple, a macaroni, a gelato, canned peas, a ground beef, a cheddar cheese, a sausage, canned beans, a chicken broth, a ketchup, a wooden spoon, a stove, a knife, a bowl, a frying pan, a whisk, a cutting board
- **Instruction**: Make scrambled eggs with cheddar cheese using the frying pan
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `CLEAR` / `CLEAR`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: Eggs, cheddar cheese and the frying pan are all present and each occurs once, so the action chain is complete. All three strategies answer CLEAR stably, showing that when ingredients, container and tool can each be grounded in the environment the model does not invent a clarification request.

### Case 3 - `s_26c1d58255b90b1694ac`

- **Pair**: test_261 | **Category**: preferences | **Gold label**: CLEAR
- **Environment**: paper towels, a plastic food storage container, a glass food storage container, a metal bowl, a plastic bowl, a ceramic bowl, honey, apple pie, key lime pie, cream pie
- **Instruction**: Clean the ceramic bowl with paper towels.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `CLEAR` / `CLEAR`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: Plastic, metal and ceramic bowls all exist in the environment, but the instruction restricts the referent to the ceramic bowl and paper towels are unique. What decides clarity is whether the referent is uniquely determined, not how many similar objects the scene holds. All three strategies are correct.

### Case 4 - `s_02f7e32fa52a6d864f19`

- **Pair**: test_7 | **Category**: preferences | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: a plastic dinner plate, a plastic bread plate, a plastic salad plate, a ceramic dinner plate, a ceramic bread plate, a ceramic salad plate, a glass dinner plate, a glass bread plate, a glass salad plate, a metal bowl, a plastic bowl, a ceramic bowl, a spatula, a milk chocolate tablet, an almond milk chocolate tablet, a dark chocolate tablet, a glass milk bottle, an oat milk bottle, black tea bags, green tea bags, mozzarella sticks, a fresh mozzarella package, cream cheese, cottage cheese
- **Instruction**: Robot, please melt the chocolate tablet in the microwave using a ceramic bowl, then mix it with a spoonful of cream cheese in the same bowl until smooth.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **Reading**: The environment holds three chocolate tablets (milk, almond milk, dark), so "the chocolate tablet" has no unique referent: a genuine referential ambiguity. All three strategies answer NEEDS_CLARIFICATION in both rounds, which makes non-unique reference the ambiguity type these models detect most reliably.

### Case 5 - `s_0a10f5bfaa0422a15fe0`

- **Pair**: test_883 | **Category**: safety | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: walnuts, a white rice, peanuts, cashews, a jasmine rice, almonds, a mixing bowl, a pot holder, a baking sheet, a food processor, a timer, a spatula
- **Instruction**: Make a mixed nut snack by roasting peanuts, almonds, walnuts, and cashews in the oven until done
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **Reading**: Neither oven temperature nor duration is given, and "until done" offers no executable stopping rule: a parameter ambiguity. All three strategies are correct. Compare case 8 (test 842), where 350F and 15 minutes are stated, to see how parameter executability drives the decision.

## 5.2 Typical errors

### Case 6 - `s_0881fd0f539f7a8abdd3`

- **Pair**: test_314 | **Category**: preferences | **Gold label**: CLEAR
- **Environment**: a knife block, a vegetable peeler, a potato masher, mixed fruit jam, a milk chocolate tablet, a dark chocolate tablet, an almond milk chocolate tablet, honey, a tomato, a cabbage, a potato, a carrot, a garlic, a green onion, a cucumber, a bell pepper
- **Instruction**: Robot, please take the knife block from the kitchen counter, and select a sharp knife for cutting vegetables. Next, take the tomato, the cabbage, the potato, the carrot, the garlic, the green onion, the cucumber, and the bell pepper from the fridge, and wash them thoroughly using the sink and the clean sponge. Dry the vegetables using the kitchen towel, and then cut them into small, bite-sized pieces using the knife. Place the cut vegetables in a large mixing bowl, and add a tablespoon of olive oil and a pinch of salt and pepper. Mix the vegetables well, until they are evenly coated with the oil and seasonings.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **Reading**: The instruction is one long but explicit procedure, yet the environment lists only vegetables and cutting tools (knife block, tomato, cabbage, cucumber and so on) and does not contain the sink, clean sponge, kitchen towel, mixing bowl or olive oil the instruction refers to. Gold is still CLEAR, so this dataset labels an instruction clear when its text is self-contained, not when the scene can execute it. All three strategies answer NEEDS_CLARIFICATION by the second criterion. This systematic gap is a main source of the elevated unnecessary-clarification rate and shows why gold semantics need manual review (manual_semantic_review is still PENDING in data_manifest.json).

### Case 7 - `s_0b7d7474407d3bf152c6`

- **Pair**: test_628 | **Category**: common_sense_knowledge | **Gold label**: CLEAR
- **Environment**: a garlic bread, oysters, a brown rice, a fettuccine, a jasmine rice, a pita bread, bagels, lasagna sheets, a penne, a basmati rice, chicken wings, a macaroni, a spaghetti, dinner rolls, a sausage, a stove
- **Instruction**: Cook the chicken wings in the oven and boil the penne pasta on the stove.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: Gold is CLEAR, treating oven temperature and time as commonsense defaults, but the environment lists a stove and no oven, so the instruction is literally short of both a parameter and a tool: a borderline sample. Checklist is correct while Zero-shot and Few-shot are wrong, which shows the label does not coincide with scene executability and that the structured Checklist tracks the gold convention more closely.

### Case 8 - `s_2bb6ef81cbafaa6811c7`

- **Pair**: test_842 | **Category**: safety | **Gold label**: CLEAR
- **Environment**: a canned chili, walnuts, a canned corn, peanuts, cashews, a canned beef stew, a penne, a pot, a baking dish, a timer, a spoon, a bowl, oven mitts, a strainer
- **Instruction**: Make a beef and penne casserole by mixing cooked penne pasta with canned beef stew, topping with crushed walnuts, and baking it in the oven at 350°F for 15 minutes.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **Reading**: Temperature (350F), duration (15 minutes), container (baking dish) and ingredients are all stated, yet the environment lists only oven mitts and no oven, and all three strategies still answer NEEDS_CLARIFICATION. This mirrors case 5: there a missing parameter was correctly flagged, here complete parameters are still treated as ambiguous because the appliance is absent. It shows the conservative bias on baking-style instructions and the fact that a referenced object missing from the scene nearly always triggers a request.

### Case 9 - `s_012698e1f503e46799fe`

- **Pair**: test_896 | **Category**: safety | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: a sour cream, walnuts, a whipped cream, a white rice, a heavy cream, a cream cheese, a spaghetti, a brown rice, a yogurt, a brie cheese, a wooden spoon, a pot, a stove, a serving bowl, a measuring cup, a heat-resistant glove
- **Instruction**: Make creamy rice with walnuts by cooking white rice, then stirring in hot heavy cream and sour cream, and topping with chopped walnuts.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `CLEAR` / `CLEAR`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: Gold is NEEDS_CLARIFICATION because the process is underspecified: how much heavy cream and sour cream, how long to cook and how far to stir are left open, while every object and tool is present in the scene. All three strategies answer CLEAR, a clear false negative. The model checks whether objects exist but not whether the process is executable, which is exactly what drives missed-ambiguity rates of 0.46 / 0.21 / 0.44.

### Case 10 - `s_02be24721a136a1d637a`

- **Pair**: test_788 | **Category**: common_sense_knowledge | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: burger buns, a cake, a bacon, a tomato, a gelato, a popcorn, peanuts, pretzels, a beef patty, cashews, a lettuce, a sausage, a spatula, a plate, a knife, a frying pan
- **Instruction**: Cook a beef burger until done with tomato and lettuce on a burger bun.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: "until done" is the same ambiguity type as case 5, yet here two of the three strategies answer CLEAR and only Few-shot is correct. The same ambiguity is treated differently in different contexts, so the decision is not stable and a single instruction cannot be extrapolated into a strategy capability.

## 5.3 Strategy splits

### Case 11 - `s_0259b77552c8d8d6b550`

- **Pair**: test_315 | **Category**: preferences | **Gold label**: CLEAR
- **Environment**: a stainless steel tablespoon, a wooden tablespoon, a silver teaspoon, paper towels, bacon, ham, granulated sugar, table salt, sea salt, muesli, cornflakes, coconut oil, olive oil, sunflower oil, blackberries, raspberries, blueberries
- **Instruction**: Please, use the stainless steel tablespoon to transfer the blackberries, the raspberries, and the blueberries from their respective containers to a small bowl. Then, use the wooden tablespoon to mix the berries together.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: The environment holds a stainless steel tablespoon, a wooden tablespoon and a silver teaspoon, but the instruction names the stainless steel one, so the reference is unique. Few-shot answers NEEDS_CLARIFICATION while Zero-shot and Checklist answer CLEAR: Few-shot over-sensitivity turns directly into an unnecessary clarification here.

### Case 12 - `s_04247c8bffe09de638b8`

- **Pair**: test_358 | **Category**: common_sense_knowledge | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: a mixer, a whisk, mustard, mayonnaise, ketchup, a bottle of white wine, a bottle of red wine, muesli, cornflakes, a garlic, a cabbage, a carrot, an onion, a cucumber, a black pepper, an energy bar, cupboard
- **Instruction**: Kitchen Robot, please use the mixer to prepare a simple coleslaw salad by mixing shredded cabbage, grated carrot, diced onion, mayonnaise, mustard, and black pepper. Serve the coleslaw in a salad bowl.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: The instruction asks to serve the coleslaw in a salad bowl, but the environment lists a mixer, a bowl and a cupboard, and no salad bowl, so the precondition cannot be met and gold is NEEDS_CLARIFICATION. Only Few-shot catches it, which shows that the higher recall of Few-shot comes from asking whenever anything is missing, at the cost of a 66 percent unnecessary-clarification rate.

### Case 13 - `s_68fec364435b48203b05`

- **Pair**: test_210 | **Category**: preferences | **Gold label**: CLEAR
- **Environment**: a plastic food storage container, a glass food storage container, chopsticks, a stainless steel dinner fork, a stainless steel salad fork, a stainless steel dinner knife, a dark chocolate tablet, an almond milk chocolate tablet, a milk chocolate tablet, ginger biscuits, digestive biscuits, chocolate teacakes, crackers, a carrot cake, a cheesecake, a vanilla cake, a chocolate cake, a strawberry cake, an ice cream cake, a vanilla yogurt cup, a greek yogurt cup, table salt, granulated sugar
- **Instruction**: Take out the vanilla yogurt cup from the fridge and place it on the kitchen table.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: The environment holds both a vanilla yogurt cup and a greek yogurt cup, and the instruction already names the vanilla one, so the reference is unique and gold is CLEAR. Zero-shot and Few-shot are wrong while Checklist is correct: the object-reference item on the Checklist is what confirms the unique referent, making this the case where Checklist alone is right.

### Case 14 - `s_8ff8e533fd171caad6d7`

- **Pair**: test_363 | **Category**: preferences | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: a ladle, paper towels, a plastic food storage container, a glass food storage container, mussels, sardines, a seaweed, shrimps, oysters, a salmon, a coconut donut, a strawberry frosted donut, a cinnamon sugar donut, apple pie, cream pie, key lime pie, pumpkin pie, shepherd's pie, a greek yogurt cup, a vanilla yogurt cup, a strawberry yogurt cup, bottled water, a corkscrew, a Pepsi can, a beer can, a Coca-Cola can, bottled iced tea, a Sprite bottle, a bottle opener, a Red Bull can, an orange soda,  plastic dinner plate, plastic bread plate, plastic salad plate, ceramic dinner plate, ceramic bread plate, ceramic salad plate
- **Instruction**: Create a seafood platter with mussels, sardines, shrimps, oysters, and salmon.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot: `CLEAR` / `CLEAR`
  - Checklist: `CLEAR` / `CLEAR`
- **Reading**: The environment holds six plates (plastic and ceramic, dinner, bread and salad) but no platter, and neither the plate type nor the quantity is stated, so gold is NEEDS_CLARIFICATION. Only Zero-shot is correct while Few-shot and Checklist answer CLEAR, which shows the direction of disagreement is not one-sided.

### Case 15 - `s_af12a678f8b469a20963`

- **Pair**: test_86 | **Category**: preferences | **Gold label**: NEEDS_CLARIFICATION
- **Environment**: a plastic dinner plate, a plastic bread plate, a plastic salad plate, a ceramic dinner plate, a ceramic bread plate, a ceramic salad plate, a glass dinner plate, a glass bread plate, a glass salad plate, a dish rack, a plastic food storage container, a glass food storage container, bacon, ham, red seedless grapes, green seedless grapes, a bottle of red wine, a corkscrew, an almond milk chocolate tablet, a dark chocolate tablet, a milk chocolate tablet, chicken nuggets, a buffalo wing
- **Instruction**: Place the dinner plates carefully in the dishwasher.
- **Raw answers, identical in R1 and R2**:
  - Zero-shot: `CLEAR` / `CLEAR`
  - Few-shot: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist: `NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **Reading**: The environment holds plastic, ceramic and glass dinner plates and no dishwasher (only a dish rack), so the instruction is both referentially ambiguous and unexecutable, and gold is NEEDS_CLARIFICATION. Zero-shot answers CLEAR, a false negative, while Few-shot and Checklist are correct. Together with cases 12 and 13 this means the five disagreement cases cover each strategy being the only correct one.

## 5.4 Observations across the 15 cases

1. Gold and the model apply different criteria. Gold asks whether the instruction text is self-contained; the model asks whether the scene can execute it. In cases 6, 7 and 8, all labelled CLEAR, the sink and the oven mentioned by the instruction are absent from the environment, so all three strategies request clarification. This single gap explains why unnecessary clarification sits at 0.45 to 0.66, and it shows the labels need manual review rather than being treated as absolute truth.
2. The model is strong on referential uniqueness and weak on process executability. Cases 4 (three chocolate tablets) and 15 (three dinner plates, no dishwasher) are flagged as ambiguous by at least two strategies, while cases 9 (missing quantity and heat level) and 10 ("until done") are answered CLEAR by most strategies. A missing object gets detected; a missing process parameter does not.
3. Each strategy is the only correct one in at least one case. Few-shot alone is right in case 12, Checklist alone in case 13 and Zero-shot alone in case 14. The difference comes from what each prompt inspects rather than from one strategy dominating: Few-shot is conservative, Checklist relies on structured checks and Zero-shot is lenient, consistent with overall Accuracy differing by only 0.5 to 2.5 percentage points.
4. All 15 cases predict identically in both rounds, above the overall 97.67 percent round-to-round agreement (Zero-shot 100, Few-shot 95.5, Checklist 97.5). These cases are therefore stable judgement tendencies rather than sampling noise, which makes them usable as qualitative evidence.

The 15 cases split 5 success / 5 typical error / 5 strategy split, which meets the handbook requirement without any shortfall to declare.