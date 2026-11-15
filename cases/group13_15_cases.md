# Group 13 — 15 个案例分析（WANG HONGFENG）

数据来源：R1 / R2 两轮冻结预测（各 200 条 × 3 策略），标签来自 `data/processed/test_labels.csv`，
指令与环境来自 `data/processed/test_inputs.csv`；评分口径与 `src/evaluate.py` 一致。
选取规则由 `scripts/select_cases.py` 固定：按 sample_id 排序取首个匹配，15 例互不重复、分属 15 个不同的指令对，每条都可在 `predictions.jsonl` 中按 sample_id 定位。

## 5.1 成功案例

### 案例 1　`s_189912ddd83d9e90f9f3`

- **指令对**：test_613　**类别**：safety　**正确标签**：CLEAR
- **环境**：a ciabatta, a garlic bread, a jam, croissants, a nutella, dinner rolls, a sliced bread, a whole wheat bread, a toaster
- **指令**：Make toast using sliced bread in the toaster.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`CLEAR` / `CLEAR`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：环境中只有一种切片面包（a sliced bread）和一台 toaster，动作对象与器具都唯一，指令不含需要消解的成分。三种策略两轮均判 CLEAR，是"对象唯一 + 参数可执行"的最简单正例。

### 案例 2　`s_195ab566fe4c7cd3afbd`

- **指令对**：test_757　**类别**：common_sense_knowledge　**正确标签**：CLEAR
- **环境**：mussels, a canned chili, a salmon, an apple cider vinegar, a canned corn, a ham, a milk, a canned soup, a frozen yogurt, canned tomatoes, an ice cream, chicken thighs, a fettuccine, a mustard, canned peaches, a soy sauce, eggs, chicken nuggets, chicken wings, a butter, canned olives, a canned pineapple, a macaroni, a gelato, canned peas, a ground beef, a cheddar cheese, a sausage, canned beans, a chicken broth, a ketchup, a wooden spoon, a stove, a knife, a bowl, a frying pan, a whisk, a cutting board
- **指令**：Make scrambled eggs with cheddar cheese using the frying pan
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`CLEAR` / `CLEAR`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：鸡蛋、切达奶酪、frying pan 三样必需要素都在环境中且各自唯一，动作链完整。三种策略稳定判 CLEAR，说明当食材、容器、器具都能在环境里逐一落地时，模型不会凭空索要澄清。

### 案例 3　`s_26c1d58255b90b1694ac`

- **指令对**：test_261　**类别**：preferences　**正确标签**：CLEAR
- **环境**：paper towels, a plastic food storage container, a glass food storage container, a metal bowl, a plastic bowl, a ceramic bowl, honey, apple pie, key lime pie, cream pie
- **指令**：Clean the ceramic bowl with paper towels.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`CLEAR` / `CLEAR`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：环境里 plastic / metal / ceramic bowl 并存，但指令已用材质把对象限定为 ceramic bowl，paper towels 也唯一。可见决定清晰与否的是"被指对象是否唯一确定"，而非环境里同类物品的数量。三种策略均判对。

### 案例 4　`s_02f7e32fa52a6d864f19`

- **指令对**：test_7　**类别**：preferences　**正确标签**：NEEDS_CLARIFICATION
- **环境**：a plastic dinner plate, a plastic bread plate, a plastic salad plate, a ceramic dinner plate, a ceramic bread plate, a ceramic salad plate, a glass dinner plate, a glass bread plate, a glass salad plate, a metal bowl, a plastic bowl, a ceramic bowl, a spatula, a milk chocolate tablet, an almond milk chocolate tablet, a dark chocolate tablet, a glass milk bottle, an oat milk bottle, black tea bags, green tea bags, mozzarella sticks, a fresh mozzarella package, cream cheese, cottage cheese
- **指令**：Robot, please melt the chocolate tablet in the microwave using a ceramic bowl, then mix it with a spoonful of cream cheese in the same bowl until smooth.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **人工分析**：环境中有 milk / almond milk / dark 三种 chocolate tablet，"the chocolate tablet"没有唯一指代，属真正的指代歧义。三种策略两轮全部判 NEEDS_CLARIFICATION，说明指代不唯一是最容易被模型捕捉的歧义类型。

### 案例 5　`s_0a10f5bfaa0422a15fe0`

- **指令对**：test_883　**类别**：safety　**正确标签**：NEEDS_CLARIFICATION
- **环境**：walnuts, a white rice, peanuts, cashews, a jasmine rice, almonds, a mixing bowl, a pot holder, a baking sheet, a food processor, a timer, a spatula
- **指令**：Make a mixed nut snack by roasting peanuts, almonds, walnuts, and cashews in the oven until done
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **人工分析**：烘烤温度与时间均缺失，"until done"没有可执行判据，属参数歧义。三种策略稳定判对。与案例 8（test 842，明确写了 350°F / 15 分钟与 baking dish）对照，可看出"参数是否可执行"是模型判断的关键线索。

## 5.2 典型错误

### 案例 6　`s_0881fd0f539f7a8abdd3`

- **指令对**：test_314　**类别**：preferences　**正确标签**：CLEAR
- **环境**：a knife block, a vegetable peeler, a potato masher, mixed fruit jam, a milk chocolate tablet, a dark chocolate tablet, an almond milk chocolate tablet, honey, a tomato, a cabbage, a potato, a carrot, a garlic, a green onion, a cucumber, a bell pepper
- **指令**：Robot, please take the knife block from the kitchen counter, and select a sharp knife for cutting vegetables. Next, take the tomato, the cabbage, the potato, the carrot, the garlic, the green onion, the cucumber, and the bell pepper from the fridge, and wash them thoroughly using the sink and the clean sponge. Dry the vegetables using the kitchen towel, and then cut them into small, bite-sized pieces using the knife. Place the cut vegetables in a large mixing bowl, and add a tablespoon of olive oil and a pinch of salt and pepper. Mix the vegetables well, until they are evenly coated with the oil and seasonings.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **人工分析**：指令是一整段完整流程，但环境清单里只有蔬菜、刀具与部分器具（knife block、番茄、卷心菜、黄瓜……），并未列出指令提到的 sink、clean sponge、kitchen towel、mixing bowl 与 olive oil。gold 仍标 CLEAR，说明该数据集的 gold 以"指令文本本身是否无歧义"为准，而不是"环境是否覆盖所有被提及的物件"。三种策略统一判 NEEDS_CLARIFICATION，正是按后者判断的结果——这类系统性差异是多余澄清率偏高的主要来源，也提示 gold 语义需要人工复核（`data_manifest.json` 中的 manual_semantic_review 仍为 PENDING）。

### 案例 7　`s_0b7d7474407d3bf152c6`

- **指令对**：test_628　**类别**：common_sense_knowledge　**正确标签**：CLEAR
- **环境**：a garlic bread, oysters, a brown rice, a fettuccine, a jasmine rice, a pita bread, bagels, lasagna sheets, a penne, a basmati rice, chicken wings, a macaroni, a spaghetti, dinner rolls, a sausage, a stove
- **指令**：Cook the chicken wings in the oven and boil the penne pasta on the stove.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：gold 标为 CLEAR（烤箱温度与时间被视作可凭常识补全），但环境里只有 stove，没有 oven，字面上既缺参数也缺器具，属边界样本。Checklist 判对，Zero-shot 与 Few-shot 判错。本例说明标签语义并不等于环境可执行性，而 Checklist 的结构化检查更贴近 gold 口径。

### 案例 8　`s_2bb6ef81cbafaa6811c7`

- **指令对**：test_842　**类别**：safety　**正确标签**：CLEAR
- **环境**：a canned chili, walnuts, a canned corn, peanuts, cashews, a canned beef stew, a penne, a pot, a baking dish, a timer, a spoon, a bowl, oven mitts, a strainer
- **指令**：Make a beef and penne casserole by mixing cooked penne pasta with canned beef stew, topping with crushed walnuts, and baking it in the oven at 350°F for 15 minutes.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **人工分析**：温度（350°F）、时长（15 分钟）、容器（baking dish）与原料都写明，但环境里只列了 oven mitts，没有 oven 本身，三种策略仍全判 NEEDS_CLARIFICATION。与案例 5 构成镜像：案例 5 缺参数被判对，本例参数齐全却因器具缺失被判歧义，反映模型对"烘焙/温度"类指令的保守倾向，同时也说明被提及物件未出现在环境中时，模型几乎必然索要澄清。

### 案例 9　`s_012698e1f503e46799fe`

- **指令对**：test_896　**类别**：safety　**正确标签**：NEEDS_CLARIFICATION
- **环境**：a sour cream, walnuts, a whipped cream, a white rice, a heavy cream, a cream cheese, a spaghetti, a brown rice, a yogurt, a brie cheese, a wooden spoon, a pot, a stove, a serving bowl, a measuring cup, a heat-resistant glove
- **指令**：Make creamy rice with walnuts by cooking white rice, then stirring in hot heavy cream and sour cream, and topping with chopped walnuts.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`CLEAR` / `CLEAR`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：gold 为 NEEDS_CLARIFICATION，原因是过程参数不确定：用多少 heavy cream 与 sour cream、煮多久、搅到何种程度都未给出——本例的对象与器具在环境中均齐备，缺的只是过程参数。三种策略却全判 CLEAR，是典型漏检：模型只核对了"对象是否存在"，没有核对"过程是否可执行"。这正是漏检率 0.46 / 0.21 / 0.44 的成因。

### 案例 10　`s_02be24721a136a1d637a`

- **指令对**：test_788　**类别**：common_sense_knowledge　**正确标签**：NEEDS_CLARIFICATION
- **环境**：burger buns, a cake, a bacon, a tomato, a gelato, a popcorn, peanuts, pretzels, a beef patty, cashews, a lettuce, a sausage, a spatula, a plate, a knife, a frying pan
- **指令**：Cook a beef burger until done with tomato and lettuce on a burger bun.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**："until done"与案例 5 属同一歧义类型，但这里三种策略中有两种判 CLEAR（仅 Few-shot 判对）。同一歧义在不同上下文中被区别对待，说明模型判断不稳定，单条指令的表现不能外推为策略能力。

## 5.3 策略分歧

### 案例 11　`s_0259b77552c8d8d6b550`

- **指令对**：test_315　**类别**：preferences　**正确标签**：CLEAR
- **环境**：a stainless steel tablespoon, a wooden tablespoon, a silver teaspoon, paper towels, bacon, ham, granulated sugar, table salt, sea salt, muesli, cornflakes, coconut oil, olive oil, sunflower oil, blackberries, raspberries, blueberries
- **指令**：Please, use the stainless steel tablespoon to transfer the blackberries, the raspberries, and the blueberries from their respective containers to a small bowl. Then, use the wooden tablespoon to mix the berries together.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：环境里同时有 stainless steel / wooden tablespoon 与 silver teaspoon，但指令已指定不锈钢汤匙，指代唯一。Few-shot 判 NEEDS_CLARIFICATION，Zero-shot 与 Checklist 判 CLEAR——Few-shot 的过度敏感在此直接产生多余澄清。

### 案例 12　`s_04247c8bffe09de638b8`

- **指令对**：test_358　**类别**：common_sense_knowledge　**正确标签**：NEEDS_CLARIFICATION
- **环境**：a mixer, a whisk, mustard, mayonnaise, ketchup, a bottle of white wine, a bottle of red wine, muesli, cornflakes, a garlic, a cabbage, a carrot, an onion, a cucumber, a black pepper, an energy bar, cupboard
- **指令**：Kitchen Robot, please use the mixer to prepare a simple coleslaw salad by mixing shredded cabbage, grated carrot, diced onion, mayonnaise, mustard, and black pepper. Serve the coleslaw in a salad bowl.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：指令要求"serve the coleslaw in a salad bowl"，但环境里只有 mixer、bowl、cupboard 等，没有 salad bowl，执行前提不满足故 gold 为 NEEDS_CLARIFICATION。三种策略中仅 Few-shot 命中，说明 Few-shot 的高召回来自"对任何缺项都发问"，代价是 66% 的多余澄清率。

### 案例 13　`s_68fec364435b48203b05`

- **指令对**：test_210　**类别**：preferences　**正确标签**：CLEAR
- **环境**：a plastic food storage container, a glass food storage container, chopsticks, a stainless steel dinner fork, a stainless steel salad fork, a stainless steel dinner knife, a dark chocolate tablet, an almond milk chocolate tablet, a milk chocolate tablet, ginger biscuits, digestive biscuits, chocolate teacakes, crackers, a carrot cake, a cheesecake, a vanilla cake, a chocolate cake, a strawberry cake, an ice cream cake, a vanilla yogurt cup, a greek yogurt cup, table salt, granulated sugar
- **指令**：Take out the vanilla yogurt cup from the fridge and place it on the kitchen table.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：环境里 vanilla yogurt cup 与 greek yogurt cup 并存，指令已指名 vanilla，指代唯一，gold 为 CLEAR。Zero-shot 与 Few-shot 判错、Checklist 判对——Checklist 中"对象指代"这一检查项帮助它确认了唯一指代，是本批案例中 Checklist 独占正确的代表。

### 案例 14　`s_8ff8e533fd171caad6d7`

- **指令对**：test_363　**类别**：preferences　**正确标签**：NEEDS_CLARIFICATION
- **环境**：a ladle, paper towels, a plastic food storage container, a glass food storage container, mussels, sardines, a seaweed, shrimps, oysters, a salmon, a coconut donut, a strawberry frosted donut, a cinnamon sugar donut, apple pie, cream pie, key lime pie, pumpkin pie, shepherd's pie, a greek yogurt cup, a vanilla yogurt cup, a strawberry yogurt cup, bottled water, a corkscrew, a Pepsi can, a beer can, a Coca-Cola can, bottled iced tea, a Sprite bottle, a bottle opener, a Red Bull can, an orange soda,  plastic dinner plate, plastic bread plate, plastic salad plate, ceramic dinner plate, ceramic bread plate, ceramic salad plate
- **指令**：Create a seafood platter with mussels, sardines, shrimps, oysters, and salmon.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Few-shot：`CLEAR` / `CLEAR`
  - Checklist：`CLEAR` / `CLEAR`
- **人工分析**：环境有 plastic / ceramic 共 6 种 plate，却没有 platter，且未说明用哪种盘子与份量，gold 为 NEEDS_CLARIFICATION。仅 Zero-shot 判对，Few-shot 与 Checklist 判 CLEAR——本例是 Zero-shot 独占正确的代表，说明策略分歧的方向并不单向。

### 案例 15　`s_af12a678f8b469a20963`

- **指令对**：test_86　**类别**：preferences　**正确标签**：NEEDS_CLARIFICATION
- **环境**：a plastic dinner plate, a plastic bread plate, a plastic salad plate, a ceramic dinner plate, a ceramic bread plate, a ceramic salad plate, a glass dinner plate, a glass bread plate, a glass salad plate, a dish rack, a plastic food storage container, a glass food storage container, bacon, ham, red seedless grapes, green seedless grapes, a bottle of red wine, a corkscrew, an almond milk chocolate tablet, a dark chocolate tablet, a milk chocolate tablet, chicken nuggets, a buffalo wing
- **指令**：Place the dinner plates carefully in the dishwasher.
- **三种策略原始回答（R1 / R2 两轮一致）**：
  - Zero-shot：`CLEAR` / `CLEAR`
  - Few-shot：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
  - Checklist：`NEEDS_CLARIFICATION` / `NEEDS_CLARIFICATION`
- **人工分析**：环境有 plastic / ceramic / glass 三种 dinner plate，且不存在 dishwasher（只有 dish rack），既是指代歧义也是不可执行指令，gold 为 NEEDS_CLARIFICATION。Zero-shot 判 CLEAR 属漏检，Few-shot 与 Checklist 判对，与案例 12、13 一起说明本批五个分歧案例覆盖了三种策略各自独占正确的情形。

## 5.4 跨案例观察

1. **gold 口径与模型口径不同**：gold 以"指令文本是否自足"为准，模型以"环境能否执行"为准。案例 6、7、8 三条 gold 为 CLEAR 的样本，指令提到的 sink、oven 都不在环境清单里，因此三种策略一致索要澄清。这一条同时解释了多余澄清率为何处在 0.45–0.66，也说明 gold 标签需要人工复核，不能当作绝对真值。
2. **模型强于"指代唯一性"，弱于"过程可执行性"**：案例 4（三种 chocolate tablet）与案例 15（三种 dinner plate 且无 dishwasher）都被至少两种策略识别为歧义；而案例 9（缺用量与火候）、案例 10（until done）却被多数策略判为 CLEAR。也就是说，缺对象能发现，缺过程参数发现不了。
3. **三种策略各自都有独占正确的情形**：案例 12 仅 Few-shot 判对、案例 13 仅 Checklist 判对、案例 14 仅 Zero-shot 判对。差异来自判断依据不同——Few-shot 偏保守、Checklist 依赖结构检查、Zero-shot 偏宽松——而不是某一策略全面更优，这与总体 Accuracy 只相差 0.5–2.5 个百分点一致。
4. **15 例在两轮执行中预测完全一致**，高于总体 97.67% 的两轮一致率（Zero-shot 100%、Few-shot 95.5%、Checklist 97.5%），说明这些案例不是随机波动的产物，而是稳定的判断倾向，适合作为定性证据引用。

以上 15 例的数量分布为 5 成功 / 5 典型错误 / 5 策略分歧，满足手册要求，未出现需要缩量说明的情形。