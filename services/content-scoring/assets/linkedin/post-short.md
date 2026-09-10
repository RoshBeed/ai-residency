We break monoliths apart for a living.

I spent a week finding out that in machine learning, the same instinct quietly costs you accuracy.

Here's the setup.

I built a model to predict how many upvotes a Hacker News post gets. Two inputs: the title, and everything else — when you posted, where it links, who you are.

Two ways to build it.

MODULAR
One model scores the title. Another scores the rest. Add the two numbers.
Separate owners. Separate deploys. One falls over, you still get an answer.

MONOLITHIC
One model sees everything at once.
Harder to run. But it can learn things the modular version structurally cannot.

Now, let's be honest about our side of the fence first.

We don't split monoliths for speed. A network hop is slower than a function call.

We split them for deploys, ownership, blast radius. And we pay in latency — and in anything that crosses a boundary. Joins get hard. Transactions get hard.

Machine learning charges exactly the same toll. Different currency.

Split the model and you lose whatever needed both halves at once. Not "learns it worse". Cannot represent it. At any size.

I tested that directly: giving the modular version eight times the capacity recovered nine percent of the gap. It's a wall, not a hill.

On Hacker News, that toll is real. The monolith wins.

And the biggest gap is between the title and when you posted it.

Here's why — the whole post in one line:

A good title doesn't add upvotes. It multiplies them.

Post something great at 3am and it's worth a handful.
Post the same thing when the site is awake and it's worth hundreds.

The modular version can only ever ADD its two scores together. So it can't express "this title, at this time". It has to average across both, and be wrong in both directions.

One rule, both fields: cut along the seams.

Nobody puts a service boundary through tightly coupled logic and enjoys the result. Titles and timing are tightly coupled. There is no seam there.

The good news is that this is measurable before you commit.

Build both. See whether the monolith actually wins. It takes an afternoon, and it tells you whether you're allowed the architecture you'd rather operate.

I went in assuming modular would be fine. It wasn't, and the reason was one multiplication.

Where have you seen a boundary drawn in the wrong place?
