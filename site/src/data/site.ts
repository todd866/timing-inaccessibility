import {
          Activity, BookOpen, BrainCircuit, ClipboardList, Database, FileDown, Gauge, GitCompareArrows, Library, Milestone, Network, Radar, Route, Sparkles, Waves
        } from "lucide-react";

        const icons = {
          Activity, BookOpen, BrainCircuit, ClipboardList, Database, FileDown, Gauge, GitCompareArrows, Library, Milestone, Network, Radar, Route, Sparkles, Waves
        };

        export const paper = {
          title: "Timing Inaccessibility and the Projection Bound",
          subtitle: "Resolving Maxwell's demon for continuous biological substrates",
          journal: "BioSystems",
          manuscriptId: "doi:10.1016/j.biosystems.2025.105632",
          decision: "Published",
          dueDate: "2025",
          doi: "10.1016/j.biosystems.2025.105632",
          abstract: "Continuous biological systems can defer irreversible projection; timing information becomes available only when a system pays to register it.",
          shortThesis: "Continuous biological systems can defer irreversible projection; timing information becomes available only when a system pays to register it.",
          mark: "T",
          pdfLabel: "PDF",
        };

        export const navLinks = [
          { href: "/", label: "Manuscript", icon: BookOpen },
          { href: "/dashboard", label: "Overview", icon: Radar },
          { href: "/revision", label: "Map", icon: ClipboardList },
          { href: "/audit", label: "Audit", icon: Gauge },
          { href: "/references", label: "References", icon: Library },
        ];

        export const downloads = [
  {
    "href": "/paper.pdf",
    "label": "PDF",
    "icon": "FileDown"
  },
  {
    "href": "/paper-source.tex",
    "label": "Source TeX",
    "icon": "FileDown"
  },
  {
    "href": "/citation-use-audit.md",
    "label": "Citation audit",
    "icon": "Database"
  },
  {
    "href": "/references.bib",
    "label": "Bibliography",
    "icon": "Library"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const headlineMetrics = [
  {
    "label": "Publication",
    "value": "BioSystems 258",
    "detail": "Article 105632 (2025)",
    "tone": "primary"
  },
  {
    "label": "Paper DOI",
    "value": "105632",
    "detail": "10.1016/j.biosystems.2025.105632",
    "tone": "tertiary"
  },
  {
    "label": "Cited sources",
    "value": "33",
    "detail": "32 bibliography entries parsed from TeX",
    "tone": "secondary"
  },
  {
    "label": "PDF/text coverage",
    "value": "25/32",
    "detail": "3 DOI-backed entries still need PaperLibrary harvest",
    "tone": "tertiary"
  }
];
        export const stanceCards = [
  {
    "title": "Two-stage model",
    "body": "Reversible correlation precedes irreversible projection, shifting the thermodynamic cost to the registration boundary.",
    "icon": "GitCompareArrows"
  },
  {
    "title": "Temporal registration",
    "body": "Ordering events requires stabilized temporal records; the cost scales with the number of orderings made explicit.",
    "icon": "Route"
  },
  {
    "title": "Projection bound",
    "body": "Collapsing a high-dimensional accessible set into a lower-dimensional record dissipates the lost distinguishability.",
    "icon": "Gauge"
  },
  {
    "title": "Camera-engine duality",
    "body": "A substrate can compute by evolving physically, while a camera-like observer pays to discretize the outcome.",
    "icon": "Activity"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const revisionPriorities = [
  {
    "title": "Separate detection from erasure",
    "reviewer": "Paper architecture",
    "priority": "Critical",
    "body": "The paper distinguishes channel-detection limits from the cost of irreversible memory stabilization."
  },
  {
    "title": "Generalize Maxwell's demon",
    "reviewer": "Paper architecture",
    "priority": "High",
    "body": "The demon problem is moved from discrete bits to continuous biological path degeneracy."
  },
  {
    "title": "Quantify timing loss",
    "reviewer": "Paper architecture",
    "priority": "High",
    "body": "Temporal order is treated as a record that must be created, not freely read off."
  },
  {
    "title": "Connect artificial systems",
    "reviewer": "Paper architecture",
    "priority": "Medium",
    "body": "The argument explains when analog and neuromorphic systems gain by deferring discretization."
  }
];
        export const stressTests = [
  {
    "title": "Two-stage model",
    "body": "Reversible correlation precedes irreversible projection, shifting the thermodynamic cost to the registration boundary.",
    "icon": "GitCompareArrows"
  },
  {
    "title": "Temporal registration",
    "body": "Ordering events requires stabilized temporal records; the cost scales with the number of orderings made explicit.",
    "icon": "Route"
  },
  {
    "title": "Projection bound",
    "body": "Collapsing a high-dimensional accessible set into a lower-dimensional record dissipates the lost distinguishability.",
    "icon": "Gauge"
  },
  {
    "title": "Camera-engine duality",
    "body": "A substrate can compute by evolving physically, while a camera-like observer pays to discretize the outcome.",
    "icon": "Activity"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const empiricalTests = [
  {
    "title": "Projection boundary",
    "body": "Where does the system first stabilize a reusable, communicable record?"
  },
  {
    "title": "Temporal resolution",
    "body": "How many temporal orderings must be distinguished for the observer's claim to hold?"
  },
  {
    "title": "Dimensional collapse",
    "body": "How many accessible pre-projection states are collapsed into the registered outcome?"
  }
];
        export const evidenceClusters = [
  {
    "title": "Thermodynamic information",
    "keys": [
      "landauer1961",
      "bennett1982",
      "szilard1929",
      "sagawa2010"
    ],
    "icon": "Gauge"
  },
  {
    "title": "Continuous demons",
    "keys": [
      "parrondo1996",
      "vaikuntanathan2009",
      "allahverdyan2009"
    ],
    "icon": "Sparkles"
  },
  {
    "title": "Metric and temporal bounds",
    "keys": [
      "kolmogorov1959",
      "slepian1961",
      "sivak2012"
    ],
    "icon": "Route"
  },
  {
    "title": "Biological timing substrates",
    "keys": [
      "anastassiou2011",
      "stringer2019",
      "todd2025biosystems"
    ],
    "icon": "Waves"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const reviewerPosture = [
  {
    "reviewer": "Published article",
    "stance": "BioSystems",
    "summary": "This native site renders the full published manuscript for Timing Inaccessibility as web text with live citations and local PDF exports."
  },
  {
    "reviewer": "Reference layer",
    "stance": "PaperLibrary-backed",
    "summary": "Each cited key receives a reference page with manuscript contexts, DOI links where available, and the current local PDF/text harvest state."
  }
];
        export const reframes = [
  {
    "before": "The demon pays for measurement.",
    "after": "The demon pays when measurement is stabilized into reusable memory."
  },
  {
    "before": "Timing is passively observed.",
    "after": "Timing is created as a record at the projection boundary."
  },
  {
    "before": "Simulation simply predicts the substrate.",
    "after": "A predictor must often emulate the substrate up to the same collapse points."
  }
];
        export const detailHighlights = [
  {
    "id": "main-thesis",
    "phrase": "Continuous biological systems can defer irreversible projection; timing information becomes available only when a system pays to register it.",
    "title": "Main thesis",
    "summary": "Continuous biological systems can defer irreversible projection; timing information becomes available only when a system pays to register it.",
    "bullets": [
      "Reversible correlation precedes irreversible projection, shifting the thermodynamic cost to the registration boundary.",
      "Ordering events requires stabilized temporal records; the cost scales with the number of orderings made explicit.",
      "Collapsing a high-dimensional accessible set into a lower-dimensional record dissipates the lost distinguishability."
    ]
  }
];
