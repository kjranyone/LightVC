# Attribution of non-speech sources used in training

The gate and mask networks of the front end (models/front_gate) were trained on mixtures of generated speech with real non-speech audio
(music, noise, hand and object sounds). Only sources under CC0, Public Domain or CC BY (without NC / ND / SA) were used, and clips with
detected speech were removed (AudioSet classifier MIT/ast-finetuned-audioset-10-10-0.4593, threshold 0.02). No source audio is distributed.
This list covers every source in the selection ledgers (a superset of what each training run drew from).

## MUSAN (D. Snyder, G. Chen, D. Povey, 2015; openslr.org/17)

Music and noise: 703 files (543 public domain / CC0, 160 CC BY).
Full credits for each file are in the LICENSE files of the MUSAN corpus. CC BY files used:

| file | license / credit (as recorded in MUSAN) |
|---|---|
| music-fma-0003 | “Memory Bank” (by Airglow) “Electrifying Landscape” (by Airglow) “Far Apart” (by Airglow) “Cepheid Disk” (by Airglow) “Blueshift” (by Airglow) “Lisa” (by Airglow) Memory Bank by Airglow is licensed un |
| music-fma-0029 | “Feels Like Home” (by Fabian Measures) “Higekitekina (悲劇的な)” (by Fabian Measures) “Subtle Library” (by Fabian Measures) “Hanami (花見)” (by Fabian Measures) “Melody of The Strings” (by Fabian Measures) |
| music-fma-0030 | “Feels Like Home” (by Fabian Measures) “Higekitekina (悲劇的な)” (by Fabian Measures) “Subtle Library” (by Fabian Measures) “Hanami (花見)” (by Fabian Measures) “Melody of The Strings” (by Fabian Measures) |
| music-fma-0031 | “Feels Like Home” (by Fabian Measures) “Higekitekina (悲劇的な)” (by Fabian Measures) “Subtle Library” (by Fabian Measures) “Hanami (花見)” (by Fabian Measures) “Melody of The Strings” (by Fabian Measures) |
| music-fma-0032 | “Feels Like Home” (by Fabian Measures) “Higekitekina (悲劇的な)” (by Fabian Measures) “Subtle Library” (by Fabian Measures) “Hanami (花見)” (by Fabian Measures) “Melody of The Strings” (by Fabian Measures) |
| music-fma-0033 | “Feels Like Home” (by Fabian Measures) “Higekitekina (悲劇的な)” (by Fabian Measures) “Subtle Library” (by Fabian Measures) “Hanami (花見)” (by Fabian Measures) “Melody of The Strings” (by Fabian Measures) |
| music-fma-0059 | “12 MORNINGS” (by Jason Shaw) “ACOUSTIC BLUES” (by Jason Shaw) “HOEDOWN” (by Jason Shaw) “WORDS” (by Jason Shaw) Audionautix: Acoustic by Jason Shaw is licensed under a Attribution 3.0 United States L |
| music-fma-0060 | “12 MORNINGS” (by Jason Shaw) “ACOUSTIC BLUES” (by Jason Shaw) “HOEDOWN” (by Jason Shaw) “WORDS” (by Jason Shaw) Audionautix: Acoustic by Jason Shaw is licensed under a Attribution 3.0 United States L |
| music-fma-0061 | “12 MORNINGS” (by Jason Shaw) “ACOUSTIC BLUES” (by Jason Shaw) “HOEDOWN” (by Jason Shaw) “WORDS” (by Jason Shaw) Audionautix: Acoustic by Jason Shaw is licensed under a Attribution 3.0 United States L |
| music-fma-0072 | “Fight the Sea” (by Josh Woodward) “The Pawnbroker's Stepdaughter” (by Josh Woodward) “Cherry” (by Josh Woodward) “Shower Scene (Instrumental Version)” (by Josh Woodward) “Can't Take Our Love Away” (b |
| music-fma-wa-0055 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0057 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0058 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0059 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0060 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0061 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0062 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0063 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0064 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0066 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0067 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0068 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0069 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-fma-wa-0070 | Classical Sampler by Kevin MacLeod is licensed under a Creative Commons Attribution License. http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler |
| music-hd-0039 | License: CC BY 3.0 |
| music-jamendo-0044 | infradrums komodovegan lensflares morphium drill optical spasm male magnetics by Bod Attribution 3.0 Unported (CC BY 3.0) https://www.jamendo.com/en/list/a85814/male-magnetics |
| music-rfm-0000 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0002 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0003 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0004 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0005 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0006 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0008 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0009 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0010 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0012 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0013 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0014 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0015 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0018 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0021 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0022 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0024 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0025 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0026 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0027 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0028 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0032 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0035 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0040 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0041 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0043 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0045 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0048 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0049 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0051 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0053 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0056 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0058 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0059 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0060 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0062 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0065 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0066 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0067 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0069 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0070 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0072 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0073 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0074 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0080 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0083 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0084 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0085 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0087 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0089 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0090 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0091 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0092 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0093 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0095 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0101 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0102 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0103 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0106 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0110 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0112 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0113 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0117 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0118 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0119 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0121 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0123 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0127 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0128 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0130 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0131 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0134 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0136 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0138 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0139 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0140 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0141 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0143 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0145 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| music-rfm-0146 | CC BY 3.0 http://incompetech.com/music/royalty-free/index.html?genre=Blues |
| noise-sound-bible-0001 | License: Attribution 3.0 |
| noise-sound-bible-0002 | License: Attribution 3.0 |
| noise-sound-bible-0005 | License: Attribution 3.0 |
| noise-sound-bible-0008 | License: Attribution 3.0 |
| noise-sound-bible-0009 | License: Attribution 3.0 |
| noise-sound-bible-0010 | License: Attribution 3.0 |
| noise-sound-bible-0012 | License: Attribution 3.0 |
| noise-sound-bible-0013 | License: Attribution 3.0 |
| noise-sound-bible-0014 | License: Attribution 3.0 |
| noise-sound-bible-0018 | License: Attribution 3.0 |
| noise-sound-bible-0019 | License: Attribution 3.0 |
| noise-sound-bible-0020 | License: Attribution 3.0 |
| noise-sound-bible-0022 | License: Attribution 3.0 |
| noise-sound-bible-0023 | License: Attribution 3.0 |
| noise-sound-bible-0026 | License: Attribution 3.0 |
| noise-sound-bible-0027 | License: Attribution 3.0 |
| noise-sound-bible-0030 | License: Attribution 3.0 |
| noise-sound-bible-0033 | License: Attribution 3.0 |
| noise-sound-bible-0034 | License: Attribution 3.0 |
| noise-sound-bible-0036 | License: Attribution 3.0 |
| noise-sound-bible-0037 | License: Attribution 3.0 |
| noise-sound-bible-0038 | License: Attribution 3.0 |
| noise-sound-bible-0040 | License: Attribution 3.0 |
| noise-sound-bible-0041 | License: Attribution 3.0 |
| noise-sound-bible-0042 | License: Attribution 3.0 |
| noise-sound-bible-0046 | License: Attribution 3.0 |
| noise-sound-bible-0047 | License: Attribution 3.0 |
| noise-sound-bible-0050 | License: Attribution 3.0 |
| noise-sound-bible-0051 | License: Attribution 3.0 |
| noise-sound-bible-0053 | License: Attribution 3.0 |
| noise-sound-bible-0054 | License: Attribution 3.0 |
| noise-sound-bible-0055 | License: Attribution 3.0 |
| noise-sound-bible-0056 | License: Attribution 3.0 |
| noise-sound-bible-0058 | License: Attribution 3.0 |
| noise-sound-bible-0060 | License: Attribution 3.0 |
| noise-sound-bible-0061 | License: Attribution 3.0 |
| noise-sound-bible-0062 | License: Attribution 3.0 |
| noise-sound-bible-0063 | License: Attribution 3.0 |
| noise-sound-bible-0064 | License: Attribution 3.0 |
| noise-sound-bible-0065 | License: Attribution 3.0 |
| noise-sound-bible-0066 | License: Attribution 3.0 |
| noise-sound-bible-0067 | License: Attribution 3.0 |
| noise-sound-bible-0068 | License: Attribution 3.0 |
| noise-sound-bible-0069 | License: Attribution 3.0 |
| noise-sound-bible-0070 | License: Attribution 3.0 |
| noise-sound-bible-0071 | License: Attribution 3.0 |
| noise-sound-bible-0073 | License: Attribution 3.0 |
| noise-sound-bible-0076 | License: Attribution 3.0 |
| noise-sound-bible-0077 | License: Attribution 3.0 |
| noise-sound-bible-0078 | License: Attribution 3.0 |
| noise-sound-bible-0079 | License: Attribution 3.0 |
| noise-sound-bible-0080 | License: Attribution 3.0 |
| noise-sound-bible-0082 | License: Attribution 3.0 |
| noise-sound-bible-0085 | License: Attribution 3.0 |

Hand and object sounds from MUSAN noise: 568 files (54 CC BY; credits in MUSAN's LICENSE files).

- noise-sound-bible-0001
- noise-sound-bible-0002
- noise-sound-bible-0005
- noise-sound-bible-0008
- noise-sound-bible-0009
- noise-sound-bible-0010
- noise-sound-bible-0012
- noise-sound-bible-0013
- noise-sound-bible-0014
- noise-sound-bible-0018
- noise-sound-bible-0019
- noise-sound-bible-0020
- noise-sound-bible-0022
- noise-sound-bible-0023
- noise-sound-bible-0026
- noise-sound-bible-0027
- noise-sound-bible-0030
- noise-sound-bible-0033
- noise-sound-bible-0034
- noise-sound-bible-0036
- noise-sound-bible-0037
- noise-sound-bible-0038
- noise-sound-bible-0040
- noise-sound-bible-0041
- noise-sound-bible-0042
- noise-sound-bible-0046
- noise-sound-bible-0047
- noise-sound-bible-0050
- noise-sound-bible-0051
- noise-sound-bible-0053
- noise-sound-bible-0054
- noise-sound-bible-0055
- noise-sound-bible-0056
- noise-sound-bible-0058
- noise-sound-bible-0060
- noise-sound-bible-0061
- noise-sound-bible-0062
- noise-sound-bible-0063
- noise-sound-bible-0064
- noise-sound-bible-0065
- noise-sound-bible-0066
- noise-sound-bible-0067
- noise-sound-bible-0068
- noise-sound-bible-0069
- noise-sound-bible-0070
- noise-sound-bible-0071
- noise-sound-bible-0073
- noise-sound-bible-0076
- noise-sound-bible-0077
- noise-sound-bible-0078
- noise-sound-bible-0079
- noise-sound-bible-0080
- noise-sound-bible-0082
- noise-sound-bible-0085

## Free Music Archive (fma_small; M. Defferrard et al., 2017)

Instrumental tracks: 365 (78 public domain / CC0, 287 CC BY).

| track id | title | artist | album | license |
|---|---|---|---|---|
| 009846 | Computer Music/Post-Fordism | Mattin | Proletarian of Noise | Attribution 3.0 United States |
| 010693 | 6.6.05 | BOPD | 2005 Stayin Alive InstruMentals | Creative Commons Attribution |
| 010694 | aint no thing | BOPD | 2005 Stayin Alive InstruMentals | Creative Commons Attribution |
| 010696 | 2.18.05 dai | BOPD | 2005 Stayin Alive InstruMentals | Creative Commons Attribution |
| 010697 | 1.26 beat1 | BOPD | 2005 Stayin Alive InstruMentals | Creative Commons Attribution |
| 010699 | 2.2.05 | BOPD | 2005 Stayin Alive InstruMentals | Creative Commons Attribution |
| 014344 | Ecclesia | Ambienteer | netBloc Vol. 22: Life on Ceres | Creative Commons Attribution |
| 015471 | Brakemans Blues | Bruce Russell | 21st Century Field Hollers and Prison Songs | Attribution 3.0 United States |
| 015771 | The Staff | Multifaros | The Factory | Attribution 3.0 United States |
| 015772 | The Product | Multifaros | The Factory | Attribution 3.0 United States |
| 024418 | 6,49 | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024421 | Black Ant | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024422 | Boomerang | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024424 | fACEMELTERmp3 | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024425 | Fater Lee | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024427 | Michael Vendette | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024428 | Underdog | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 024429 | Oh K. | Black Ant | Free Beats Sel. 3 | Creative Commons Attribution |
| 030636 | connecticut | If, Bwana | Pogus/White Fungus Compilation | Attribution 3.0 United States |
| 030702 | Walk through Trees | Gurdonark | Classwar Karaoke - 0010 Survey | Creative Commons Attribution |
| 035008 | Sadness | Gurdonark | Classwar Karaoke - 0011 Survey | Creative Commons Attribution |
| 037119 | Dark Hallway | Lee Rosevere | Calling All Fiends | Attribution 3.0 International |
| 038859 | Berries | Gurdonark | Classwar Karaoke - 0012 Survey | Creative Commons Attribution |
| 042372 | Only Instrumental | Broke For Free | Directionless EP | Creative Commons Attribution |
| 042373 | Mell's Parade | Broke For Free | Directionless EP | Creative Commons Attribution |
| 042374 | My Luck | Broke For Free | Directionless EP | Creative Commons Attribution |
| 042375 | Day Bird | Broke For Free | Directionless EP | Creative Commons Attribution |
| 042376 | My Always Mood | Broke For Free | Directionless EP | Creative Commons Attribution |
| 042377 | Night Owl | Broke For Free | Directionless EP | Creative Commons Attribution |
| 045099 | Spell | Rolemusic | Straw Fields | Attribution |
| 045100 | Leafless Quince Tree | Rolemusic | Straw Fields | Attribution |
| 045101 | Straw Fields | Rolemusic | Straw Fields | Attribution |
| 045102 | Yellow Dust | Rolemusic | Straw Fields | Attribution |
| 045103 | Poppies | Rolemusic | Straw Fields | Attribution |
| 047071 | All of Us | Eric Skiff | Resistor Anthems | Creative Commons Attribution |
| 050952 | Cattails | Kevin MacLeod | Contemporary Sampler | Creative Commons Attribution |
| 050955 | Eastminster | Kevin MacLeod | Contemporary Sampler | Creative Commons Attribution |
| 050956 | Devastation and Revenge | Kevin MacLeod | Contemporary Sampler | Creative Commons Attribution |
| 051004 | Tikopia | Kevin MacLeod | Global Sampler | Creative Commons Attribution |
| 051005 | Rite of Passage | Kevin MacLeod | Global Sampler | Creative Commons Attribution |
| 051006 | Quasi Motion | Kevin MacLeod | Global Sampler | Creative Commons Attribution |
| 051111 | Motormouth | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 051112 | Arms and Legs | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 051113 | Shadow Pocket | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 051114 | Wack | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 051115 | Hymenoptera | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 051117 | Affairs of State | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 051118 | Tickled Pink | Bob Ostertag | Motormouth | Creative Commons Attribution |
| 052628 | Vanishing Horizon | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052629 | Groovy Baby | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052630 | Quirky Jerk | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052631 | Big Car Theft | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052636 | Fat Caps | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052637 | Feels Good 2 B | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052639 | Get A Move On | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052644 | Night Rave | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052646 | Pyramids | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052648 | Tech Talk | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052649 | The Big House | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 052650 | Timen Passing By | Jason Shaw | Audionautix: Tech, Urban, Dance | Attribution 3.0 United States |
| 054463 | Swan Lake Dub | DuBoLoGy | Dub Lab | Creative Commons Attribution |
| 054464 | Dub Atao | DuBoLoGy | Dub Lab | Creative Commons Attribution |
| 054467 | Chacha Dub | DuBoLoGy | Dub Lab | Creative Commons Attribution |
| 054468 | Dub Action | DuBoLoGy | Dub Lab | Creative Commons Attribution |
| 054470 | Epic Dub | DuBoLoGy | Dub Lab | Creative Commons Attribution |
| 061006 | Week One | Ben von Wildenhaus | Instrumental Quaalude, vol. 1 | Creative Commons Attribution |
| 061007 | Week Two | Ben von Wildenhaus | Instrumental Quaalude, vol. 1 | Creative Commons Attribution |
| 061011 | Week Seven | Ben von Wildenhaus | Instrumental Quaalude, vol. 1 | Creative Commons Attribution |
| 064626 | Jangle | Blank & Kytt | Heavy, Crazy, Serious | Creative Commons Attribution |
| 064856 | Cloud Fight 2 | Monroeville Music Center | Musique Concrète for Beginners | Creative Commons Attribution |
| 064857 | Rainbow Bomb | Monroeville Music Center | Musique Concrète for Beginners | Creative Commons Attribution |
| 064858 | Trigger Select | Monroeville Music Center | Musique Concrète for Beginners | Creative Commons Attribution |
| 064860 | Subterranean Centipede Slave | Monroeville Music Center | Musique Concrète for Beginners | Creative Commons Attribution |
| 064861 | Herb Pilhofer's Whirlpool | Monroeville Music Center | Musique Concrète for Beginners | Creative Commons Attribution |
| 064862 | Hairy Fairy Hotaruna | Monroeville Music Center | Musique Concrète for Beginners | Creative Commons Attribution |
| 064863 | En Route (ou Enfin! Le Défilé Capsule de Temps va au Centre de Contrôle) | Monroeville Music Center | Le Progrès | Creative Commons Attribution |
| 064864 | Préparatifs dans la Salle de Rédaction | Monroeville Music Center | Le Progrès | Creative Commons Attribution |
| 064865 | Il Pleut des Informations (ou Mass Media en Masse à Partir de Notre Cumulonimus Fiable) | Monroeville Music Center | Le Progrès | Creative Commons Attribution |
| 064866 | Á Minuit, le 31 Décembre | Monroeville Music Center | Le Progrès | Creative Commons Attribution |
| 068838 | David's Tune | Howie Mitchell | Early 70s | Attribution 3.0 United States |
| 068839 | Lucky's Tune | Howie Mitchell | Early 70s | Attribution 3.0 United States |
| 068840 | John's Laughing Place | Howie Mitchell | Early 70s | Attribution 3.0 United States |
| 071503 | @ | ui | Monorave Compo | Creative Commons Attribution |
| 071506 | Cloudy | Subway Sonicbeat | Monorave Compo | Creative Commons Attribution |
| 071508 | Burn Down the Monostery | egr | Monorave Compo | Creative Commons Attribution |
| 071510 | Chemtrails | Din Sky | Monorave Compo | Creative Commons Attribution |
| 071512 | Noam Chomsky's Corporate Carbohydrate Surprise | Quelle Fromage | Monorave Compo | Creative Commons Attribution |
| 071513 | M0NOR4V3N | XC3N | Monorave Compo | Creative Commons Attribution |
| 071515 | How Ya Like Me Now Pt.2 | BLEO | Monorave Compo | Creative Commons Attribution |
| 071822 | The Cemetery | Sinqlogue | The Haunted Mansion | Creative Commons Attribution |
| 072067 | Africa | Decktonic | Stars | Attribution |
| 072069 | Act IV | Decktonic | Stars | Attribution |
| 072070 | Variations | Decktonic | Stars | Attribution |
| 072073 | Z-Saber (feat Jake Allison) | Decktonic | Stars | Attribution |
| 072074 | Fair Game | Decktonic | Stars | Attribution |
| 072075 | Night Drive | Decktonic | Stars | Attribution |
| 074908 | In A Graveyard On Halloween Night | Did You Just Hex Me? | Did You Just Hex Me? | Creative Commons Attribution |
| 075927 | Bullit | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 075931 | King of the Streets | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 075932 | World of Concrete | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 075933 | My Love | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 075935 | Still Balling | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 075936 | Trying to Understand | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 075937 | So Much Pain | Tricky Diesel | Give Me a Beat | Creative Commons Attribution |
| 076654 | Forgotten Machines | Decktonic | Forgotten Machines | Attribution |
| 076659 | Future Music | Decktonic | Forgotten Machines | Attribution |
| 078213 | Psaltery Piece | Howie Mitchell | Howie Mitchell - Singles | Attribution 3.0 United States |
| 088861 | Cien Volando | Josh Woodward | Dirty Wings | Attribution 3.0 United States |
| 088869 | Gallows Hill (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088870 | Methylchloroisothiazolinone (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088871 | Omaha (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088872 | I Want to Destroy Something Beautiful (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088873 | Cien Volando (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088875 | But It Did (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088877 | Insomnia (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088878 | Gravity (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 088892 | The Vagabond (No Vocals) | Josh Woodward | Dirty Wings (Instrumentals) | Attribution 3.0 United States |
| 089814 | Dreams | ENDRUDARK | Dreams | Creative Commons Attribution |
| 089816 | Time Up | ENDRUDARK | Dreams | Creative Commons Attribution |
| 091158 | The Perfect Planet | Monroeville Music Center | Choose Your Own Adventure | Creative Commons Attribution |
| 091159 | The Race Forever | Monroeville Music Center | Choose Your Own Adventure | Creative Commons Attribution |
| 091160 | Invaders From Within | Monroeville Music Center | Choose Your Own Adventure | Creative Commons Attribution |
| 091161 | The Reality Machine | Monroeville Music Center | Choose Your Own Adventure | Creative Commons Attribution |
| 091162 | Motocross Mania | Monroeville Music Center | Choose Your Own Adventure | Creative Commons Attribution |
| 091164 | Roller Star | Monroeville Music Center | Choose Your Own Adventure | Creative Commons Attribution |
| 091619 | A ninja among culturachippers | Rolemusic | gig's n' contest | Creative Commons Attribution |
| 091620 | ayVentura | Rolemusic | gig's n' contest | Creative Commons Attribution |
| 091622 | Scape from the city | Rolemusic | gig's n' contest | Creative Commons Attribution |
| 091623 | Death on the battlefield | Rolemusic | gig's n' contest | Creative Commons Attribution |
| 091624 | Donsi Killed la rioja star (Familiar live) | Rolemusic | gig's n' contest | Creative Commons Attribution |
| 091625 | l3go | Rolemusic | gig's n' contest | Creative Commons Attribution |
| 093518 | Mantilla | Moon Veil | Snake Charm | Creative Commons Attribution |
| 093519 | Scorpio | Moon Veil | Snake Charm | Creative Commons Attribution |
| 093523 | Up There | Moon Veil | Snake Charm | Creative Commons Attribution |
| 106502 | Hanami (花見) | Fabian Measures | Hanami | Attribution |
| 107425 | The Spidrman's (nanoloop) | UncleBibby | Free Music Project Vol. 1 | Attribution |
| 107426 | Chunky Worm Stew | UncleBibby | Free Music Project Vol. 1 | Attribution |
| 107432 | The Simple Complex | UncleBibby | Free Music Project Vol. 1 | Attribution |
| 107434 | Bubbling Pots and Troubling Thoughts | UncleBibby | Free Music Project Vol. 1 | Attribution |
| 107435 | Jangus (inspired by ToeJam and Earl) | UncleBibby | Free Music Project Vol. 1 | Attribution |
| 107579 | Abiding | Kelly Latimore | Trails | Attribution |
| 107584 | Ode to Diving Pelicans | Kelly Latimore | Unaccountable odes | Attribution |
| 107592 | Prelude | Kelly Latimore | Prelude | Attribution |
| 107596 | Armitage | Kelly Latimore | Prelude | Attribution |
| 108415 | AstroTurf | UncleBibby | Free Music Project Vol. 2 | Attribution |
| 108416 | HomeBaser | UncleBibby | Free Music Project Vol. 2 | Attribution |
| 108418 | sandwork | UncleBibby | Free Music Project Vol. 2 | Attribution |
| 108421 | Sunshiner (TWOPLUSFADE mix) | UncleBibby | Free Music Project Vol. 2 | Attribution |
| 108422 | YearBear | UncleBibby | Free Music Project Vol. 2 | Attribution |
| 108423 | Heavin' On Earth | UncleBibby | Free Music Project Vol. 2 | Attribution |
| 108906 | Did you know? (Curiouser and curiouser) | Fabian Measures | Singles Album | Attribution |
| 111819 | Ad Astra | P C III | Ad Astra, Vol. 1 | Attribution |
| 112315 | Digital Lightning | Cloudkicker | Little histories | Attribution |
| 112316 | Hassan | Cloudkicker | Little histories | Attribution |
| 112317 | Signal/Noise | Cloudkicker | Little histories | Attribution |
| 113110 | The Tragedy of Loss | Ian Alex Mac | Cues (Additional Tracks) | Attribution |
| 113970 | Clover | Little Glass Men | Spontaneous Album Sampler | Attribution |
| 113971 | Dream Blaze | Little Glass Men | Spontaneous Album Sampler | Attribution |
| 113972 | Your Pulse | Little Glass Men | Spontaneous Album Sampler | Attribution |
| 114212 | Old Fashioned Lullaby | Squire Tuck | Collection | Attribution |
| 114222 | Ukulele Blues | Squire Tuck | Collection | Attribution |
| 114223 | Ukulele Rising | Squire Tuck | Collection | Attribution |
| 114242 | Disco Time | Tentacles | Dance | Attribution |
| 114268 | poison ivy yard work | Uncle Milk | Route 9 | Attribution |
| 114272 | Travelor _.sinh remix | Uncle Milk | Route 9 | Attribution |
| 114274 | Kalaarmari | Uncle Milk | Route 9 | Attribution |
| 114385 | Black Radiations | Quaro | EP | Attribution |
| 114386 | Lost | Quaro/Bruno Del Ben | EP | Attribution |
| 114389 | Nidra | Quaro | EP | Attribution |
| 114390 | Poison | Quaro | EP | Attribution |
| 114391 | Psion | Quaro | EP | Attribution |
| 114392 | Summon | Quaro | EP | Attribution |
| 114393 | Yojimbo | Quaro | EP | Attribution |
| 114396 | Breathe New Life | Scott Holmes | Colossal Stillness (2014) | Attribution |
| 114398 | Uninvited | Scott Holmes | Colossal Stillness (2014) | Attribution |
| 114402 | Remember The Archer | Scott Holmes | Colossal Stillness (2014) | Attribution |
| 114403 | Close To The Distance | Scott Holmes | Colossal Stillness (2014) | Attribution |
| 114405 | Reminiscence | Scott Holmes | Colossal Stillness (2014) | Attribution |
| 114408 | Feather | Scott Holmes | We Used To Paint Stars In The Sky (2012) | Attribution |
| 114558 | delete_me | epsilon not | dayvision | Attribution |
| 115321 | Desert Castles | Little Glass Men | Spontaneous Album Sampler | Attribution |
| 115339 | Whiteout | Little Glass Men | Spontaneous Album Sampler | Attribution |
| 115767 | The Last Whale | Jelsonic | Various Moods | Attribution |
| 115768 | Lost Star | Jelsonic | Various Moods | Attribution |
| 115769 | Moonsea | Jelsonic | Various Moods | Attribution |
| 115771 | Saying Goodbye In The Rain (piano) | Jelsonic | Various Moods | Attribution |
| 115772 | The Returning | Jelsonic | Various Moods | Attribution |
| 115775 | Vapour Trails | Jelsonic | Various Moods | Attribution |
| 115811 | OST 01 Tutorial | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 115813 | OST 03 Ship | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 115814 | OST 04 Ship under attack | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 115815 | OST 05 Go Go Go | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 115816 | OST 06 Go Go Faster Faster | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 116877 | 8-bit Love Machine | Gumbel | 8-bit Love Machine | Attribution |
| 116878 | Tapes | Gumbel | 8-bit Love Machine | Attribution |
| 116879 | El Siniestro Y La Velóz | Gumbel | 8-bit Love Machine | Attribution |
| 116881 | Killing Bosses | Gumbel | 8-bit Love Machine | Attribution |
| 118500 | 3 | Jonathan Zorn | 2-10 | Attribution |
| 118618 | Hanami Matsuri (花見祭り) | Fabian Measures | Hanami | Attribution |
| 118920 | Golden | Little Glass Men | Debut E.P. | Attribution |
| 118921 | Westside Chillers | Little Glass Men | Debut E.P. | Attribution |
| 118922 | Modulation of the Spirit | Little Glass Men | Debut E.P. | Attribution 3.0 United States |
| 122671 | The Whisper | P C III | Ad Astra, Vol. 2 | Attribution |
| 123964 | 5-11-15 | BenJamin Banger | #JustBecause | Attribution |
| 123965 | Freestlye 30 | BenJamin Banger | #JustBecause | Attribution |
| 123966 | Bruce Wayne | BenJamin Banger | #JustBecause | Attribution |
| 123968 | Michael Van Allen | BenJamin Banger | #JustBecause | Attribution |
| 123970 | Agoge | BenJamin Banger | #JustBecause | Attribution |
| 123971 | Freestyle 32 | BenJamin Banger | #JustBecause | Attribution |
| 123972 | Awkward Romance | BenJamin Banger | #JustBecause | Attribution |
| 123973 | Ric Flair 1.5 | BenJamin Banger | #JustBecause | Attribution |
| 123974 | Everything New (2015) | BenJamin Banger | #JustBecause | Attribution |
| 123977 | Freestyle 33 | BenJamin Banger | #JustBecause | Attribution |
| 123979 | Moments | BenJamin Banger | #JustBecause | Attribution |
| 123980 | Ric Flair 2 | BenJamin Banger | #JustBecause | Attribution |
| 124154 | не осуждайте Гитлера (он болен) ... | Kosta T | ALLBIOM | Attribution |
| 124175 | Breeeze | Jelsonic | Various Moods II | Attribution |
| 124176 | Lost Longing | Jelsonic | Various Moods II | Attribution |
| 124177 | Darkling Skies | Jelsonic | Various Moods II | Attribution |
| 124178 | Seasong | Jelsonic | Various Moods II | Attribution |
| 124179 | Another Brilliant Age | Jelsonic | Various Moods II | Attribution |
| 124183 | Saying Goodbye In The Rain (string) | Jelsonic | Various Moods II | Attribution |
| 124231 | AVIATION CRAB ORCHESTRA (PACHELBELS CANNON) | ODOR BABY | AVIATION CRAB ORCHESTRA (AN EPIC SAGA) | Attribution |
| 124873 | Crystal Echoes | Sergey Cheremisinov | Sea & Night | Attribution |
| 124891 | Proyecto U-90 | Gabriel Pereira Spurr | La vida secreta del Coronel Kurtz | Attribution |
| 126300 | Scramby Eggs | Andy G. Cohen | Layers (EP) | Attribution |
| 126646 | LifeCycle | Fabian Measures | Hanami | Attribution |
| 127263 | OST 08 Rescued :-) | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 127996 | Labyrinth | Sergey Cheremisinov | Forgotten Stars | Attribution |
| 128758 | Here Come The Horses | Synapsis | Armageddon | Attribution |
| 130129 | Artificial anxiety | Michett | Palettes | Attribution |
| 130130 | Atmo | Michett | Palettes | Attribution |
| 130132 | Droids in a bar | Michett | Palettes | Attribution |
| 130133 | Minora | Michett | Palettes | Attribution |
| 130134 | Oceanic | Michett | Palettes | Attribution |
| 130135 | Picnic at the reactor | Michett | Palettes | Attribution |
| 130136 | Wet | Michett | Palettes | Attribution |
| 131837 | 『鯖江市日の出』Sabae Sunrise | Fabian Measures | Singles Album | Attribution |
| 132134 | OST 12 Wherever Aliens!!! | sawsquarenoise | RottenMage SpaceJacked OST | Attribution |
| 132310 | Twinkle Twinkle (Instrumental) | David Mumford | Bonfire Music | Attribution |
| 132448 | Liberosis (Instrumental) | Nihilore | Misophonia | Attribution |
| 134586 | Now You Are Here | Sergey Cheremisinov | Northern City's Secret | Attribution |
| 134922 | -Shab Ayum | Turku, Nomads of the Silk Road | Nomads of the Silk Road | Attribution |
| 134928 | -Uskudara Gideriken | Turku, Nomads of the Silk Road | Alleys of Istanbul | Attribution |
| 134935 | -Yesilim | Turku, Nomads of the Silk Road | Alleys of Istanbul | Attribution |
| 134936 | -Drum Solo | Turku, Nomads of the Silk Road | Alleys of Istanbul | Attribution |
| 134937 | -Muhabet | Turku, Nomads of the Silk Road | Alleys of Istanbul | Attribution |
| 134941 | -Bir Demet Yasemen | Turku, Nomads of the Silk Road | Alleys of Istanbul | Attribution |
| 136331 | Delirium | Kai Engel | ICD-10 | Attribution |
| 137054 | Oh My God | S-B-J | Singles | Attribution |
| 139776 | Unveiling Soul | Evgeny Teilor | Appearances | Attribution |
| 140925 | Set Fire to Creation | Nihilore | The Hermeneutic Circle | Creative Commons Attribution |
| 141144 | Land Legs | Andy G. Cohen | MUL / DIV | Attribution |
| 141590 | Dub Mirror | Mikuś | Dub Mirror | Attribution |
| 141591 | From Ever Since | Mikuś | Dub Mirror | Attribution |
| 141593 | Temple Ball | Mikuś | Dub Mirror | Attribution |
| 141873 | Refinery | Remain | Remain | Attribution |
| 141878 | II | Nocturnal Mayhem | NM | Attribution |
| 142079 | OOOTHERSIDE - Crystal Air | Mike B. Fort | OOOTHERSIDE | Attribution |
| 142080 | OOOTHERSIDE - Liquid Paper | Mike B. Fort | OOOTHERSIDE | Attribution |
| 142666 | Floridian Fighting Bear | J1* | The Longest Short Ends | Attribution |
| 142668 | Fox 103 | J1* | The Longest Short Ends | Attribution |
| 142669 | Mello Hello | J1* | The Longest Short Ends | Attribution |
| 142670 | Crispy Roses | J1* | The Longest Short Ends | Attribution |
| 142671 | Silverdirt | J1* | The Longest Short Ends | Attribution |
| 142672 | Fireheart | J1* | The Longest Short Ends | Attribution |
| 143055 | Robot Heart | Mystery Mammal | Wonders Of Modern Technology | Attribution |
| 143059 | Ugh! Humans Can Join | Mystery Mammal | Wonders Of Modern Technology | Attribution |
| 143060 | Wonders Of Modern Technology | Mystery Mammal | Wonders Of Modern Technology | Attribution |
| 144492 | New Moon | Dave Keifer | Howdy Persephone | Attribution |
| 144809 | Что хочешь... | Kosta T | ALLBIOM | Attribution |
| 144810 | SickNik( | Kosta T | ALLBIOM | Attribution |
| 145002 | A Dancer Stole My Heart | Finn Anderson | Uncharted Lands EP | Attribution |
| 145431 | Open Cab | Fabian Measures | Singles Album | Attribution |
| 146020 | Arvet | Ars Sonor & Malga Kubiak | Raoul Wallenbergs Ömhet (En Hjälte Eller En Spion?) | Attribution |
| 147780 | Dimension | Creo | Dimension | Attribution |
| 149185 | Touch You | Mike B. Fort | Touch You | Attribution |
| 149775 | Beyond the Looking Glass | Controlled Insanity | Into the Pit | Attribution |
| 149777 | Daqaah | Controlled Insanity | Into the Pit | Attribution |
| 149778 | Inner Mental Cage (Bitmap Remix) | Controlled Insanity | Into the Pit | Attribution |
| 150267 | Olivia | Hyson | Soundtrack for the Weary Vol. II | Attribution |
| 150268 | All Who Are Weary | Hyson | Soundtrack for the Weary Vol. II | Attribution |
| 154303 | In It to Win It | Fleslit | Journey | Attribution |
| 154305 | Calm Flight | Fleslit | Journey | Attribution |
| 154307 | 12:01 AM | Fleslit | Journey | Attribution |
| 154308 | MIA | Fleslit | Journey | Attribution |
| 154309 | A1 Symphony | Fleslit | Journey | Attribution |

## FSD50K (development split; E. Fonseca et al., 2022; Zenodo 4060432)

Hand and object sounds: 1512 clips (880 CC0, 632 CC BY). Each clip is a Freesound sound: https://freesound.org/s/<id>/

| freesound id | title | uploader | license |
|---|---|---|---|
| 137 | keyboard-typing.wav | Anton | CC BY |
| 3536 | record scratch.wav | luffy | CC BY |
| 3666 | scrat001.wav | NoiseCollector | CC BY |
| 3668 | scratch1.wav | NoiseCollector | CC BY |
| 3669 | scratch98.wav | NoiseCollector | CC BY |
| 3670 | scratch99.wav | NoiseCollector | CC BY |
| 3672 | start002.wav | NoiseCollector | CC BY |
| 3673 | start003.wav | NoiseCollector | CC BY |
| 6231 | zipper001.wav | Anton | CC BY |
| 6232 | zipper002.wav | Anton | CC BY |
| 6233 | zipper003.wav | Anton | CC BY |
| 6234 | zipper004.wav | Anton | CC BY |
| 7742 | zipper2.wav | NoiseCollector | CC BY |
| 7743 | zipper3.wav | NoiseCollector | CC BY |
| 7765 | Handclaps.wav | NoiseCollector | CC BY |
| 12660 | Zipper_01.wav | mich3d | CC BY |
| 12661 | Zipper_HandSaw_01.wav | mich3d | CC BY |
| 14579 | typeing2.ogg | bjornredtail | CC BY |
| 18314 | beef jerky bag soft clip1.wav | luffy | CC BY |
| 18316 | beef jerky bag.wav | luffy | CC BY |
| 18317 | beef jerky bag1.wav | luffy | CC BY |
| 18323 | shakejerkybag.wav | luffy | CC BY |
| 18324 | stereojerkybag.wav | luffy | CC BY |
| 18325 | stereojerkybag1.wav | luffy | CC BY |
| 18326 | tearing beef jerky.wav | luffy | CC BY |
| 18728 | Zipper05.wav | LG | CC BY |
| 20195 | Heater 1.wav | stinkhorn | CC BY |
| 21104 | rive papir 4.wav | sagosen | CC BY |
| 21179 | Clap1.aif | ramjac | CC BY |
| 21180 | Clap2.aif | ramjac | CC BY |
| 21181 | Clap3.aif | ramjac | CC BY |
| 21182 | Clap4.aif | ramjac | CC BY |
| 21183 | Clap5.aif | ramjac | CC BY |
| 21592 | Mego079_1.wav | ashassin | CC BY |
| 21762 | Opening Crackers and Crunching One.mp3 | digifishmusic | CC BY |
| 22807 | echo chamber.flac | Corsica_S | CC BY |
| 26552 | clp01.wav | altemark | CC BY |
| 26554 | clp03.wav | altemark | CC BY |
| 26732 | Zipper06.wav | LG | CC BY |
| 26733 | Zipper07.wav | LG | CC BY |
| 28222 | Clap1.wav | HerbertBoland | CC BY |
| 28224 | Clap11.wav | HerbertBoland | CC BY |
| 28225 | Clap12.wav | HerbertBoland | CC BY |
| 28226 | Clap2.wav | HerbertBoland | CC BY |
| 28227 | Clap3.wav | HerbertBoland | CC BY |
| 28228 | Clap4.wav | HerbertBoland | CC BY |
| 28229 | Clap5.wav | HerbertBoland | CC BY |
| 28230 | Clap6.wav | HerbertBoland | CC BY |
| 28232 | Clap8.wav | HerbertBoland | CC BY |
| 28233 | Clap9.wav | HerbertBoland | CC BY |
| 29938 | record_scratch_short.wav | Halleck | CC BY |
| 29941 | scratch11.wav | junggle | CC BY |
| 29943 | scratch57.wav | junggle | CC BY |
| 31817 | malexmedia_frantic_typing.wav | malexmedia | CC BY |
| 32253 | scratch344_looped.wav | junggle | CC BY |
| 32254 | scratch345_looped.wav | junggle | CC BY |
| 32296 | door knock 2.wav | ERH | CC BY |
| 32297 | door knock 3.wav | ERH | CC BY |
| 32298 | door knock.wav | ERH | CC BY |
| 37166 | candy wrapper crinkle A.wav | volivieri | CC BY |
| 37167 | candy wrapper crinkle B.wav | volivieri | CC BY |
| 37168 | candy wrapper crinkle C.wav | volivieri | CC BY |
| 37170 | candy wrapper crinkle E.wav | volivieri | CC BY |
| 37171 | candy wrapper crinkle F.wav | volivieri | CC BY |
| 37172 | candy wrapper crinkle big A.wav | volivieri | CC BY |
| 37173 | candy wrapper crinkle big B.wav | volivieri | CC BY |
| 37174 | candy wrapper crinkle big C.wav | volivieri | CC BY |
| 37175 | candy wrapper crinkle big D.wav | volivieri | CC BY |
| 37176 | candy wrapper crinkle short A.wav | volivieri | CC BY |
| 37177 | candy wrapper crinkle short B.wav | volivieri | CC BY |
| 37179 | candy wrapper crinkle short D.wav | volivieri | CC BY |
| 37558 | Tapping Guitar.wav | digifishmusic | CC BY |
| 42929 | PA microphone feedback (1) tapping.wav | FreqMan | CC BY |
| 45511 | dough hook.flac | Corsica_S | CC BY |
| 46020 | crinkle.flac | Corsica_S | CC BY |
| 46077 | HandclapsLargeSpace.mp3 | HerbertBoland | CC BY |
| 46279 | 20080103.tearing.01.wav | dobroide | CC BY |
| 46280 | 20080103.tearing.04.wav | dobroide | CC BY |
| 46281 | 20080103.tearing.05.wav | dobroide | CC BY |
| 46282 | 20080103.tearing.06.wav | dobroide | CC BY |
| 46283 | 20080103.tearing.07.wav | dobroide | CC BY |
| 46342 | gear.test.figernail.tapping.wav | dobroide | CC BY |
| 57855 | tap01.wav | ebison | CC BY |
| 61856 | scratch10.wav | NoiseCollector | CC BY |
| 61872 | scratch25.wav | NoiseCollector | CC BY |
| 61874 | scratch27.wav | NoiseCollector | CC BY |
| 61876 | scratch29.wav | NoiseCollector | CC BY |
| 61877 | scratch3.wav | NoiseCollector | CC BY |
| 61879 | scratch31.wav | NoiseCollector | CC BY |
| 61880 | scratch32.wav | NoiseCollector | CC BY |
| 61881 | scratch33.wav | NoiseCollector | CC BY |
| 61889 | scratch40.wav | NoiseCollector | CC BY |
| 61895 | scratch46.wav | NoiseCollector | CC BY |
| 61896 | scratch47.wav | NoiseCollector | CC BY |
| 61897 | scratch48.wav | NoiseCollector | CC BY |
| 61899 | scratch5.wav | NoiseCollector | CC BY |
| 61904 | scratch54.wav | NoiseCollector | CC BY |
| 61906 | scratch56.wav | NoiseCollector | CC BY |
| 61913 | scratch62.wav | NoiseCollector | CC BY |
| 61914 | scratch63.wav | NoiseCollector | CC BY |
| 61924 | scratch9.wav | NoiseCollector | CC BY |
| 62748 | beat_yoyo.wav | NoiseCollector | CC BY |
| 62750 | beatout.wav | NoiseCollector | CC BY |
| 62753 | chirpple.wav | NoiseCollector | CC BY |
| 62755 | crabseizure.wav | NoiseCollector | CC BY |
| 62756 | cuebert.wav | NoiseCollector | CC BY |
| 62760 | faderattack.wav | NoiseCollector | CC BY |
| 62761 | faderofftime.wav | NoiseCollector | CC BY |
| 62769 | warped_vinylbeat_3bars.wav | NoiseCollector | CC BY |
| 62772 | warpedjungleloop.wav | NoiseCollector | CC BY |
| 63865 | 20081130.zipper.wav | dobroide | CC BY |
| 66813 | Tape Wad.aif | SunnySideSound | CC BY |
| 67786 | Paper Tearing.wav | SunnySideSound | CC BY |
| 68698 | HandClap.wav | mikaelfernstrom | CC BY |
| 68702 | KnockingDoor.wav | mikaelfernstrom | CC BY |
| 68727 | SingleHandClap.wav | mikaelfernstrom | CC BY |
| 68732 | TearingPaper.wav | mikaelfernstrom | CC BY |
| 68752 | ZipOnBag.wav | mikaelfernstrom | CC BY |
| 68892 | BAGRIPl1.wav | aboe | CC BY |
| 68893 | BAGRIPl2.wav | aboe | CC BY |
| 68894 | BAGRIPl3.wav | aboe | CC BY |
| 68895 | BAGRIPl4.wav | aboe | CC BY |
| 68896 | BAGRIPl5.wav | aboe | CC BY |
| 68897 | BAGRIPm1.wav | aboe | CC BY |
| 68898 | BAGRIPs1.wav | aboe | CC BY |
| 68899 | BAGRIPs2.wav | aboe | CC BY |
| 68900 | BAGRIPs3.wav | aboe | CC BY |
| 68901 | BAGRIPs4.wav | aboe | CC BY |
| 68902 | BAGRIPs5.wav | aboe | CC BY |
| 68904 | BAGRIPs7.wav | aboe | CC BY |
| 69209 | one person clapping.wav | FreqMan | CC BY |
| 69289 | Crinkling Plastic.wav | ZuluT | CC BY |
| 71435 | Tearing tape 01 090420.wav | LG | CC BY |
| 71436 | Tearing tape 02 090420.wav | LG | CC BY |
| 73083 | Hit basement 01.wav | LG | CC BY |
| 73086 | Hit basement 04.wav | LG | CC BY |
| 73087 | Hit basement 05.wav | LG | CC BY |
| 73088 | Hit basement 06.wav | LG | CC BY |
| 75374 | GTA4 clap.wav | jobro | CC BY |
| 76779 | Hand Clap.mp3 | DJ Burnham | CC BY |
| 76813 | door knock 2.mp3 | DJ Burnham | CC BY |
| 77326 | Keyboard01.wav | Kyster | CC BY |
| 78952 | typing.wav | bsung88 | CC BY |
| 79136 | subway sandwich wrapper.wav | bsung88 | CC BY |
| 79148 | Zipper.wav | Razzvio | CC BY |
| 79155 | Inflated ziploc bag crumpling.wav | atha89 | CC BY |
| 79161 | Unzipping a zipper.wav | atha89 | CC BY |
| 86661 | FINGER SNAPS LOOP 8BARS 100BPM.wav | sandyrb | CC BY |
| 88851 | gpt01.aif | lsprice | CC BY |
| 88855 | gpt05.aif | lsprice | CC BY |
| 88856 | gpt06.aif | lsprice | CC BY |
| 88858 | gpt08.aif | lsprice | CC BY |
| 88901 | gt02.aif | lsprice | CC BY |
| 88910 | gt11.aif | lsprice | CC BY |
| 88912 | gt13.aif | lsprice | CC BY |
| 88914 | gt15.aif | lsprice | CC BY |
| 88915 | gt16.aif | lsprice | CC BY |
| 88951 | losnb01.aif | lsprice | CC BY |
| 88952 | losnb02.aif | lsprice | CC BY |
| 88953 | losnb03.aif | lsprice | CC BY |
| 88954 | losnb04.aif | lsprice | CC BY |
| 88955 | losnb05.aif | lsprice | CC BY |
| 88957 | losnb07.aif | lsprice | CC BY |
| 88958 | losnb08.aif | lsprice | CC BY |
| 89385 | Lettuce_Frozen7_Tear.wav | zimbot | CC BY |
| 89578 | Store_Crinkling Bag1.mp3 | CGEffex | CC BY |
| 89579 | Store_Crinkling Bag2.mp3 | CGEffex | CC BY |
| 89580 | Store_Crinkling Bag3.mp3 | CGEffex | CC BY |
| 89778 | tape02-wide_painters_tape#1.flac | zerolagtime | CC BY |
| 89789 | paper02-crumpling_paper#1.flac | zerolagtime | CC BY |
| 89791 | paper02-tearing_paper#1.flac | zerolagtime | CC BY |
| 89859 | pt01.aif | lsprice | CC BY |
| 89862 | pt04.aif | lsprice | CC BY |
| 89866 | pt08.aif | lsprice | CC BY |
| 89868 | soh01.aif | lsprice | CC BY |
| 89869 | soh02.aif | lsprice | CC BY |
| 89870 | soh03.aif | lsprice | CC BY |
| 89871 | soh04.aif | lsprice | CC BY |
| 89872 | soh05.aif | lsprice | CC BY |
| 89873 | soh06.aif | lsprice | CC BY |
| 89874 | soh07.aif | lsprice | CC BY |
| 89875 | soh08.aif | lsprice | CC BY |
| 89877 | soh10.aif | lsprice | CC BY |
| 89878 | soh11.aif | lsprice | CC BY |
| 89907 | sot01.aif | lsprice | CC BY |
| 89908 | sot02.aif | lsprice | CC BY |
| 89909 | sot03.aif | lsprice | CC BY |
| 89910 | sot04.aif | lsprice | CC BY |
| 89911 | sot05.aif | lsprice | CC BY |
| 89912 | sot06.aif | lsprice | CC BY |
| 89913 | sot07.aif | lsprice | CC BY |
| 89914 | sot08.aif | lsprice | CC BY |
| 89915 | sot09.aif | lsprice | CC BY |
| 89916 | sot10.aif | lsprice | CC BY |
| 89917 | sot11.aif | lsprice | CC BY |
| 89919 | sot13.aif | lsprice | CC BY |
| 89923 | sot17.aif | lsprice | CC BY |
| 89924 | sot18.aif | lsprice | CC BY |
| 89925 | sot19.aif | lsprice | CC BY |
| 91465 | typing sound.wav | seanas80 | CC BY |
| 91506 | big_zippers.aif | Corsica_S | CC BY |
| 91507 | small_zippers.aif | Corsica_S | CC BY |
| 93135 | Hacking at a keyboard.wav | CGEffex | CC BY |
| 94994 | suitcasezipper_khaoss_616.wav | Khaoss | CC BY |
| 99967 | Tapping on large window 120bpm.wav | CGEffex | CC BY |
| 100019 | jacket zip.aif | limwei | CC BY |
| 106982 | casezipper.wav | plingativator | CC BY |
| 107448 | crinklestereoloop.wav | NoiseCollector | CC BY |
| 107449 | crunchyloop.wav | NoiseCollector | CC BY |
| 109755 | Glass tap.wav | Printer Killer | CC BY |
| 110554 | clap_with_reverb.wav | jacobvalenta | CC BY |
| 111105 | tearing paper_01.wav | Dymewiz | CC BY |
| 111106 | tearing paper_02.wav | Dymewiz | CC BY |
| 111107 | tearing paper_03.wav | Dymewiz | CC BY |
| 111108 | tearing paper_04.wav | Dymewiz | CC BY |
| 111109 | tearing paper_05.wav | Dymewiz | CC BY |
| 111110 | tearing paper_06.wav | Dymewiz | CC BY |
| 111111 | tearing paper_07.wav | Dymewiz | CC BY |
| 111112 | tearing paper_08.wav | Dymewiz | CC BY |
| 111113 | tearing paper_09.wav | Dymewiz | CC BY |
| 111114 | tearing paper_10.wav | Dymewiz | CC BY |
| 111115 | tearing paper_11.wav | Dymewiz | CC BY |
| 111116 | tearing paper_12.wav | Dymewiz | CC BY |
| 111117 | tearing paper_13.wav | Dymewiz | CC BY |
| 111119 | tearing paper_15.wav | Dymewiz | CC BY |
| 111120 | tearing paper_16.wav | Dymewiz | CC BY |
| 114041 | FollowedThuds5.wav | EdgardEdition | CC BY |
| 114042 | FollowedThuds6.wav | EdgardEdition | CC BY |
| 116754 | rustlingadollar_.wav | 1jmorrisoncafe291 | CC BY |
| 116755 | rustlingadollar_proc(1).wav | 1jmorrisoncafe291 | CC BY |
| 116756 | rustlingadollar_studio(fade).wav | 1jmorrisoncafe291 | CC BY |
| 116757 | rustlingadollar_studio.wav | 1jmorrisoncafe291 | CC BY |
| 118049 | RKeaton_EMF366_12_Tearing Thick Paper.aif | rkeato | CC BY |
| 119753 | Page Tears.wav | jesabat | CC BY |
| 120330 | knock on glass.wav | cemagar | CC BY |
| 120331 | knock on wood.wav | cemagar | CC BY |
| 120347 | slow clap.wav | cemagar | CC BY |
| 130037 | fingers_snapping.wav | MaxDemianAGL | CC BY |
| 131218 | Opening-and-closing-snap-fasteners-on-jacket.wav | sound_ims | CC BY |
| 131226 | Zipping-and-unzipping-jacket.wav | sound_ims | CC BY |
| 132411 | Tearing_rotten_wood_5.wav | AlienXXX | CC BY |
| 133657 | Lo-Fi Keyboard Typing | PaulBannisterAudio | CC BY |
| 135632 | Zip Up.WAV | PrimeJunt | CC BY |
| 135633 | Paper Rip.WAV | PrimeJunt | CC BY |
| 135644 | Paper Rip Short.WAV | PrimeJunt | CC BY |
| 135645 | Paper Rip Long.WAV | PrimeJunt | CC BY |
| 138323 | Wooden_Cane_Knocking_on_Door_FourTimes.wav | Tdude9000 | CC BY |
| 138407 | Angry Mouse clicking and typing after pc breakdown.wav | goranbrezac | CC BY |
| 139334 | aplauso.mp3 | stephiechanel | CC BY |
| 140618 | rip paper 2012-1-4.wav | Kyster | CC BY |
| 140891 | WrappingPaper1 wrap_home_Jan2012.aif | jgeralyn | CC BY |
| 140895 | WrappingPaper2 unwrap_home_Jan2012.aif | jgeralyn | CC BY |
| 144105 | typing_press_delete.wav | jwestbury | CC BY |
| 144106 | typing_phrase.wav | jwestbury | CC BY |
| 145123 | Bag-Hard_Plastic-Zipper-Unzip-01.wav | D W | CC BY |
| 146786 | Ripping_big_paper.wav | HollowRiku | CC BY |
| 150985 | Dare12_2_handy.wav | OSH37 | CC BY |
| 150986 | Dare12_1_zip.wav | OSH37 | CC BY |
| 151214 | Click / Finger Snap | OwlStorm | CC BY |
| 151216 | Crumpling Plastic | OwlStorm | CC BY |
| 151219 | Solo Clap | OwlStorm | CC BY |
| 151231 | Crumpling Paper | OwlStorm | CC BY |
| 152514 | mySound_GPSUK_flag.wav | thecityrings | CC BY |
| 153444 | Angry fast Mac keyboard typing.mp3 | orginaljun | CC BY |
| 156103 | SonicSnap_GPSUK_bag texture.wav | thecityrings | CC BY |
| 157013 | Zipper Foley | Slave2theLight | CC BY |
| 158057 | keyboard typing old | Eelke | CC BY |
| 159083 | Rip 11.aif | duncanlewismackinnon | CC BY |
| 159084 | Rip 10.aif | duncanlewismackinnon | CC BY |
| 159085 | Rip 1.aif | duncanlewismackinnon | CC BY |
| 159086 | Rip 4.aif | duncanlewismackinnon | CC BY |
| 159087 | Rip 3.aif | duncanlewismackinnon | CC BY |
| 159088 | Rip 2.aif | duncanlewismackinnon | CC BY |
| 159089 | Rip 13.aif | duncanlewismackinnon | CC BY |
| 159090 | Rip 6.aif | duncanlewismackinnon | CC BY |
| 159091 | Rip 5.aif | duncanlewismackinnon | CC BY |
| 159092 | Rip 8.aif | duncanlewismackinnon | CC BY |
| 159094 | Rip 9.aif | duncanlewismackinnon | CC BY |
| 160130 | Tunnel 2 Impulse-Response reverb clap afar high-pitch clap | jorickhoofd | CC BY |
| 160131 | Tunnel 1 Impulse-Response reverb clap closeby high-pitch clap | jorickhoofd | CC BY |
| 160134 | Tunnel at trainstation Impulse-Response reverb clap | jorickhoofd | CC BY |
| 160135 | Tunnel 2 Impulse-Response reverb closeby high-pitch clap | jorickhoofd | CC BY |
| 160137 | Staircase Impulse-Response long reverb 15m away clap | jorickhoofd | CC BY |
| 160138 | Staircase Impulse-Response long reverb 20m away clap | jorickhoofd | CC BY |
| 160140 | Viaduct Impulse-Response reverb 5m away clap | jorickhoofd | CC BY |
| 160141 | Basement Impulse-Response reverb 5m away medium-pitched clap | jorickhoofd | CC BY |
| 160143 | Basement Impulse-Response reverb 6m away medium-pitched clap | jorickhoofd | CC BY |
| 160144 | Train station tunnel Impulse-Response reverb 3m away high-pitched clap | jorickhoofd | CC BY |
| 160145 | Empty living-room Impulse-Response reverb 4m away clap | jorickhoofd | CC BY |
| 160146 | Tunnel 3 Impulse-Response reverb high-pitched clap | jorickhoofd | CC BY |
| 160948 | Chipclap01.wav | dragline777 | CC BY |
| 161414 | grabbing-chip-bag | CosmicEmbers | CC BY |
| 161586 | IR clap small Hexagonic Chapel layered | jorickhoofd | CC BY |
| 161587 | IR clap small Hexagonic Chapel hi-pitch | jorickhoofd | CC BY |
| 161588 | IR clap small Hexagonic Chapel lo-pitch | jorickhoofd | CC BY |
| 161839 | Balloon Tapping | antistatikk | CC BY |
| 163591 | Cash Register Print'n'Rip A2 | unfa | CC BY |
| 163592 | Cash Register Print'n'Rip A1 | unfa | CC BY |
| 163593 | Cash Register Print'n'Rip B2 | unfa | CC BY |
| 164887 | Hotel_do_Mar_2012-16_Large_Triangular_Room_Clap_IR.wav | AlienXXX | CC BY |
| 164888 | Hotel_do_Mar_2012-17_Large_Triangular_Room_Clap_IR.wav | AlienXXX | CC BY |
| 166691 | Keyboard | dersuperanton | CC BY |
| 167249 | Zip open and Close .wav | RobinScemama | CC BY |
| 170574 | Knocking on a Sliding Glass Door | Ryan_kuke | CC BY |
| 174433 | nervous_programmer.wav | Zabuhailo | CC BY |
| 174434 | nervous_programmer_keyboard_drop.wav | Zabuhailo | CC BY |
| 175345 | smack.WAV | mrawsome14 | CC BY |
| 175429 | Vocal_humming_while_tapping_cheek.aif | ajkerekes | CC BY |
| 175536 | flo x jomox p vii.wav | alphasnk | CC BY |
| 176154 | Slap Hands | retropttam | CC BY |
| 176438 | inverted_cheek_taps.aif | ajkerekes | CC BY |
| 176440 | taps_on_table_pitch_down.aif | ajkerekes | CC BY |
| 176560 | Slap Hands Manipulated | retropttam | CC BY |
| 177494 | snap_3.wav | Snapper4298 | CC BY |
| 177495 | snap_2.wav | Snapper4298 | CC BY |
| 177496 | snap_1.wav | Snapper4298 | CC BY |
| 177497 | Snap_1234.wav | Snapper4298 | CC BY |
| 177498 | snap_4.wav | Snapper4298 | CC BY |
| 177500 | snap_5.wav | Snapper4298 | CC BY |
| 177503 | snap_10.wav | Snapper4298 | CC BY |
| 177504 | snap_9.wav | Snapper4298 | CC BY |
| 177506 | snap_7.wav | Snapper4298 | CC BY |
| 178640 | VaseBeat2.flac | Zabuhailo | CC BY |
| 179181 | SNARE4.wav | Snapper4298 | CC BY |
| 179182 | SNARE5.wav | Snapper4298 | CC BY |
| 179183 | SNARE2.wav | Snapper4298 | CC BY |
| 179184 | SNARE3.wav | Snapper4298 | CC BY |
| 179186 | SNARE6.wav | Snapper4298 | CC BY |
| 179189 | HAT1.wav | Snapper4298 | CC BY |
| 180982 | kick2.wav | Snapper4298 | CC BY |
| 180986 | lid down 2.wav | Snapper4298 | CC BY |
| 180991 | spoon 3.wav | Snapper4298 | CC BY |
| 182022 | Five_Heavy_knocks_on_bedroom_door_2.wav | Omar Alvarado | CC BY |
| 182023 | Eight_knocks_on_bathroom2_door.wav | Omar Alvarado | CC BY |
| 182024 | 4_light_knocks_on_computer_room_door.mp3 | Omar Alvarado | CC BY |
| 182032 | Four_sharp_knocks_on_bedroom_door2.wav | Omar Alvarado | CC BY |
| 182033 | Five_knocks_on_closet_Door.wav | Omar Alvarado | CC BY |
| 182034 | Five_knocks_on_bathroom2_door.wav | Omar Alvarado | CC BY |
| 182039 | four_knocks_on_bedroom_door1.wav | Omar Alvarado | CC BY |
| 182041 | five_sharp_knocks_on_computer_room_door.wav | Omar Alvarado | CC BY |
| 182043 | five_knocks_on_bedroom_door1.wav | Omar Alvarado | CC BY |
| 182510 | Ice clinking in empty glass mug | TerryHancock | CC BY |
| 184566 | Zipper 1 | MrAuralization | CC BY |
| 184570 | Zipper 3 | MrAuralization | CC BY |
| 184571 | Zipper 2 | MrAuralization | CC BY |
| 186013 | typing.wav | conradts | CC BY |
| 187805 | ripping paper.mp3 | sonsdebarcelona | CC BY |
| 187954 | snap5.wav | stomachache | CC BY |
| 187956 | snap1.wav | stomachache | CC BY |
| 187958 | snap3.wav | stomachache | CC BY |
| 187959 | snap4.wav | stomachache | CC BY |
| 187960 | echoclap7.wav | stomachache | CC BY |
| 187961 | echoclap8.wav | stomachache | CC BY |
| 187966 | snap7.wav | stomachache | CC BY |
| 187967 | echoclap2.wav | stomachache | CC BY |
| 187970 | echoclap3.wav | stomachache | CC BY |
| 187972 | echoclap5.wav | stomachache | CC BY |
| 188864 | handclaps.wav | plingativator | CC BY |
| 188878 | Door Knocking 01.wav | Suprasummun | CC BY |
| 188883 | Door Knocking 02.wav | Suprasummun | CC BY |
| 190020 | Hit on wood - 2 | Aiwha | CC BY |
| 190023 | Hit on wood - 7 | Aiwha | CC BY |
| 190025 | Hit on wood - 5 | Aiwha | CC BY |
| 190026 | Theatre 3 hits | Aiwha | CC BY |
| 190027 | Hits on wood | Aiwha | CC BY |
| 190802 | clap5.wav | guillem_abello | CC BY |
| 192229 | Shower Clap - Closed.wav | Ligidium | CC BY |
| 194260 | Clap 1 - room.wav | datwilightz | CC BY |
| 194261 | Clap 1 - Room and Delay.wav | datwilightz | CC BY |
| 194264 | Clap 2 - Ultra Reverse Reverb.wav | datwilightz | CC BY |
| 194265 | Clap 2 - Ultra Reverse Reverb and Delay.wav | datwilightz | CC BY |
| 194269 | Clap 2.wav | datwilightz | CC BY |
| 194270 | clap 3 reverse reverb.wav | datwilightz | CC BY |
| 194647 | Small tin tap 4.wav | S. Dedalus | CC BY |
| 194648 | Small tin tap 3.wav | S. Dedalus | CC BY |
| 194650 | Small tin tap 7.wav | S. Dedalus | CC BY |
| 201228 | Crab.wav | Augdog | CC BY |
| 201255 | Paper Rip | empraetorius | CC BY |
| 201256 | Paper Crumple | empraetorius | CC BY |
| 201615 | Ripping_Paper.mp3 | deleted_user_3720673 | CC BY |
| 207305 | rubbing and tearing paper.WAV | thecityrings | CC BY |
| 208662 | lamaze toys made of a crinkly material | ginacollecchia | CC BY |
| 208749 | Zipper2.wav | EverHeat | CC BY |
| 210192 | Tap Oil.wav | Augdog | CC BY |
| 210802 | Chirping staircase | acs272 | CC BY |
| 211496 | Knocking_wood_door2.mp3 | Taira Komori | CC BY |
| 211497 | Knocking_iron_door5.mp3 | Taira Komori | CC BY |
| 211498 | Knocking_iron_door4.mp3 | Taira Komori | CC BY |
| 211499 | Knocking_wood_door1.mp3 | Taira Komori | CC BY |
| 211501 | Knocking_iron_door1.mp3 | Taira Komori | CC BY |
| 211502 | Knocking_iron_door2.mp3 | Taira Komori | CC BY |
| 211503 | Knocking_iron_door3.mp3 | Taira Komori | CC BY |
| 211608 | Keyboard Key Echoing | qubodup | CC BY |
| 212180 | Bag Zipper | OwlStorm | CC BY |
| 212393 | Zipper1.mp3 | Taira Komori | CC BY |
| 212522 | Keyboard.mp3 | Taira Komori | CC BY |
| 212664 | Plastic-bag crackle | mboscolo | CC BY |
| 213055 | clapping_alone02.mp3 | Taira Komori | CC BY |
| 213056 | clapping_alone01.mp3 | Taira Komori | CC BY |
| 216088 | Ripping | RICHERlandTV | CC BY |
| 216123 | zipper2.aif | hukleesyr | CC BY |
| 216126 | keyboard.aif | hukleesyr | CC BY |
| 216131 | keyboard3.aif | hukleesyr | CC BY |
| 216926 | tearing_a_cloth02.mp3 | Taira Komori | CC BY |
| 218980 | knocking GOOD A_2.aif | bcimmet | CC BY |
| 218981 | knocking GOOD A_1.aif | bcimmet | CC BY |
| 218984 | knocking GOOD A_4.aif | bcimmet | CC BY |
| 218985 | knocking GOOD A_3.aif | bcimmet | CC BY |
| 218987 | knocking GOOD A_6.aif | bcimmet | CC BY |
| 218988 | knocking GOOD A_5.aif | bcimmet | CC BY |
| 218989 | knocking GOOD_5.1 A.aif | bcimmet | CC BY |
| 219169 | clap_04.wav | Jagadamba | CC BY |
| 219927 | FingerTapsBenchLongLoop01.wav | Otakua | CC BY |
| 219932 | FingerTapsWoodShortLoop01.wav | Otakua | CC BY |
| 219933 | FingerTapsWoodLongLoop01.wav | Otakua | CC BY |
| 219934 | FingerTapsBenchShortLoop01.wav | Otakua | CC BY |
| 221750 | Uniball Pen Tapping on Paper on Glass.wav | hasbrouck | CC BY |
| 221942 | Snapping fingers | sonsdebarcelona | CC BY |
| 228365 | Clap.wav | 18hiltc | CC BY |
| 231891 | clap3.wav | Godowan | CC BY |
| 231893 | clap1.wav | Godowan | CC BY |
| 231894 | clap8.wav | Godowan | CC BY |
| 231895 | clap7.wav | Godowan | CC BY |
| 231896 | clap6.wav | Godowan | CC BY |
| 231897 | clap5.wav | Godowan | CC BY |
| 233020 | Computer Keyboard.mp3 | amorralteixe | CC BY |
| 233232 | keyboard2.wav | eduardfrigola | CC BY |
| 233233 | keyboard.wav | eduardfrigola | CC BY |
| 233396 | Keyboard.wav | laiaoreka | CC BY |
| 236798 | mySound-49GR-Farsad.wav | thecityrings | CC BY |
| 236809 | mySound-49GR-XiaoTian.wav | thecityrings | CC BY |
| 238568 | Example of the sample - Scratch 'Cratchin'! 2 with a beat.WAV | bronxio | CC BY |
| 239304 | Queneau crssement grave.wav | hubertmichel | CC BY |
| 241644 | Door03.wav | skinnytecboy | CC BY |
| 246327 | Computer_Keyboard_Spacebar.wav | kwahmah_02 | CC BY |
| 249520 | Hammer Hit 1.wav | kwahmah_02 | CC BY |
| 249522 | Hammer Hit 2.wav | kwahmah_02 | CC BY |
| 250699 | Zipper 6 | Aiwha | CC BY |
| 250700 | Zipper 7 | Aiwha | CC BY |
| 250701 | Zipper 4 | Aiwha | CC BY |
| 250702 | Zipper 5 | Aiwha | CC BY |
| 250705 | Zipper 8 | Aiwha | CC BY |
| 250715 | Zipper 3 | Aiwha | CC BY |
| 252715 | KEYBOARD, TYPING.wav | xtrgamr | CC BY |
| 255540 | SARCASTIC CLAPPING.wav | xtrgamr | CC BY |
| 257039 | klobin.wav | rtorga | CC BY |
| 257775 | Paper crumpling up.wav | xtrgamr | CC BY |
| 257788 | Typing on keyboard.wav | xtrgamr | CC BY |
| 257793 | Crowd golf clap.wav | xtrgamr | CC BY |
| 257805 | Paper rip.wav | xtrgamr | CC BY |
| 257828 | Paper.aif - GoSoundtrack | GoSoundtrack | CC BY |
| 257915 | snapping fingers 3.wav | erikschenkel | CC BY |
| 257916 | snapping fingers 2.wav | erikschenkel | CC BY |
| 257918 | snap certain times.wav | erikschenkel | CC BY |
| 261389 | Tearing fabric 1 | JakLocke | CC BY |
| 261393 | Tearing fabric 2 | JakLocke | CC BY |
| 261397 | Tearing fabric 3 | JakLocke | CC BY |
| 263541 | Cloth Rip | Dpoggioli | CC BY |
| 263543 | Unzipping A Zipper | Dpoggioli | CC BY |
| 265968 | Knocking On The Door.aiff | Item_Wars_Creation | CC BY |
| 266600 | Tap Tap.wav | SocializedArtist45 | CC BY |
| 267302 | Sound Design Class Keyboard edited.wav | formerfro | CC BY |
| 268826 | keyboardtype.flac | kwahmah_02 | CC BY |
| 268968 | Keyboard.WAV | kczub | CC BY |
| 269476 | Clean Record Scratch | kwahmah_02 | CC BY |
| 270821 | random typing.mp3 | troy1960 | CC BY |
| 274666 | Zipper card out of pocket.wav | BarkersPinhead | CC BY |
| 274814 | knocking.mp3 | BarkersPinhead | CC BY |
| 274943 | knock knock1 | theshaggyfreak | CC BY |
| 277298 | typewriter_manual_spacebar_several_times.wav | magedu | CC BY |
| 277299 | typewriter_manual_spacebar.wav | magedu | CC BY |
| 277300 | typewriter_manual_platen_01.wav | magedu | CC BY |
| 277301 | typewriter_manual_carriage_ring.wav | magedu | CC BY |
| 277302 | typewriter_manual_typing.wav | magedu | CC BY |
| 277411 | clap sample_rev.wav | xtrgamr | CC BY |
| 277721 | typewriter_electric_typing_01.wav | magedu | CC BY |
| 277722 | typewriter_electric_turn_on.wav | magedu | CC BY |
| 277723 | typewriter_electric_turn_off.wav | magedu | CC BY |
| 277724 | typewriter_electric_spacebar.wav | magedu | CC BY |
| 318926 | Tapping, Wall, A.wav | InspectorJ | CC BY |
| 318929 | Tapping, Window, Single, A.wav | InspectorJ | CC BY |
| 318930 | Tapping, Window, Multi, A.wav | InspectorJ | CC BY |
| 319943 | typewriter_electric_typing_02.wav | magedu | CC BY |
| 319944 | typewriter_electric_spacebar_several_times.wav | magedu | CC BY |
| 319945 | typewriter_electric_return.wav | magedu | CC BY |
| 320102 | keyboard type.wav | pipnone | CC BY |
| 320389 | Material Hits.wav | yadronoff | CC BY |
| 322227 | Tearing, Cardboard, Thin, A.wav | InspectorJ | CC BY |
| 323482 | Handclap 3 (+chorus + compressor) | meisterjaan | CC BY |
| 326728 | Tapping away on a smartphone | pfranzen | CC BY |
| 329052 | Keyboard and mouse | apolloaiello | CC BY |
| 329053 | Keyboard | apolloaiello | CC BY |
| 331158 | Very quick typing.wav | starinatwalls | CC BY |
| 332178 | Mashing on a computer keyboard | pfranzen | CC BY |
| 332458 | Keyboard Typing.wav | thegreatrazz | CC BY |
| 333776 | Unzipping purse, pulling out phone | pfranzen | CC BY |
| 334962 | 018_House_Keyboard2.wav | RicardoFM | CC BY |
| 334963 | 017_House_Keyboard.wav | RicardoFM | CC BY |
| 335994 | Tearing, Carpet, A.wav | InspectorJ | CC BY |
| 337626 | 2_zipper_normalized.wav | mitsuhito | CC BY |
| 339347 | typing-zoom.wav | newagesoup | CC BY |
| 341593 | clap.wav | dpsa | CC BY |
| 343426 | Typing_Smartphone_24bit_96kHz_BWF_Stereo.wav | Schacht | CC BY |
| 343427 | Typing_Smartphone_24bit_96kHz_BWF_Mono.wav | Schacht | CC BY |
| 345474 | DEMIN-EYMARD_celine_2015_2016billequitombeetroule.wav | iut_Paris8 | CC BY |
| 346786 | fingersnap11.wav | Dizzy Banjo | CC BY |
| 346787 | fingersnap10.wav | Dizzy Banjo | CC BY |
| 346788 | fingersnap1.wav | Dizzy Banjo | CC BY |
| 346789 | fingersnap0.wav | Dizzy Banjo | CC BY |
| 346790 | fingersnap15.wav | Dizzy Banjo | CC BY |
| 346791 | fingersnap14.wav | Dizzy Banjo | CC BY |
| 346792 | fingersnap13.wav | Dizzy Banjo | CC BY |
| 346793 | fingersnap12.wav | Dizzy Banjo | CC BY |
| 346794 | fingersnap17.wav | Dizzy Banjo | CC BY |
| 346795 | fingersnap16.wav | Dizzy Banjo | CC BY |
| 346796 | fingersnap4.wav | Dizzy Banjo | CC BY |
| 346797 | fingersnap5.wav | Dizzy Banjo | CC BY |
| 346798 | fingersnap27.wav | Dizzy Banjo | CC BY |
| 346799 | fingersnap3.wav | Dizzy Banjo | CC BY |
| 346800 | fingersnap8.wav | Dizzy Banjo | CC BY |
| 346802 | fingersnap6.wav | Dizzy Banjo | CC BY |
| 346803 | fingersnap7.wav | Dizzy Banjo | CC BY |
| 346804 | fingersnap20.wav | Dizzy Banjo | CC BY |
| 346805 | fingersnap2.wav | Dizzy Banjo | CC BY |
| 346806 | fingersnap19.wav | Dizzy Banjo | CC BY |
| 346807 | fingersnap18.wav | Dizzy Banjo | CC BY |
| 346808 | fingersnap24.wav | Dizzy Banjo | CC BY |
| 346809 | fingersnap23.wav | Dizzy Banjo | CC BY |
| 346810 | fingersnap22.wav | Dizzy Banjo | CC BY |
| 346811 | fingersnap21.wav | Dizzy Banjo | CC BY |
| 346812 | fingersnap26.wav | Dizzy Banjo | CC BY |
| 346813 | fingersnap25.wav | Dizzy Banjo | CC BY |
| 347319 | peel02.wav | newagesoup | CC BY |
| 349242 | Door Knock | Natty23 | CC BY |
| 349243 | Door Knock (Two Bits) | Natty23 | CC BY |
| 349244 | Two Door Knocks | Natty23 | CC BY |
| 350360 | TYPING-FLOURISH-SHORT.wav | newagesoup | CC BY |
| 359142 | Plastic Ikea Bag 02 | harveyjnz | CC BY |
| 359143 | Plastic Ikea Bag 01 | harveyjnz | CC BY |
| 364678 | Someone knocking at the front door | pfranzen | CC BY |
| 364680 | Someone POUNDING on your front door | pfranzen | CC BY |
| 365666 | WALTER_Pauline_2016_2017_KnockKnock.wav | univ_lyon3 | CC BY |
| 365902 | FAUVRE_Lola_2016_2017_DoorKnock.wav | univ_lyon3 | CC BY |
| 367085 | Vartanian_Méréthy_2016_2017_Typewriter.wav | univ_lyon3 | CC BY |
| 367129 | Knocking "shave and a haircut" on various surfaces | pfranzen | CC BY |
| 367803 | BRUNIER_Lucas_2016_2017_zip.wav | univ_lyon3 | CC BY |
| 368680 | KNOCK_KNOCK_01.wav | Mattix145 | CC BY |
| 368682 | KNOCK_KNOCK_02.wav | Mattix145 | CC BY |
| 369525 | Balloon Pop Studio (3).wav | JohanDeecke | CC BY |
| 369757 | Cardboard Tear Short 04 | AxelSpeller | CC BY |
| 369758 | Cardboard Tear Short 05 | AxelSpeller | CC BY |
| 369760 | Cardboard Tear Short 03 | AxelSpeller | CC BY |
| 370899 | door_four knocks_deep.wav | sjturia | CC BY |
| 370913 | door_three knocks_hard.wav | sjturia | CC BY |
| 370916 | door_multiple knocks_medium.wav | sjturia | CC BY |
| 370920 | door_three knocks_light.wav | sjturia | CC BY |
| 371624 | 228 - Acoustic Memory - Claps_02.wav | AcousticMemory | CC BY |
| 371625 | 226 - Acoustic Memory - SnapClap.wav | AcousticMemory | CC BY |
| 371633 | 208 - Acoustic Memory - TickTick_AluminumCan.wav | AcousticMemory | CC BY |
| 371657 | 229 - Acoustic Memory - Claps_03.wav | AcousticMemory | CC BY |
| 371661 | 237 - Acoustic Memory - DoorKnocks.wav | AcousticMemory | CC BY |
| 371663 | 219 - Acoustic Memory - Snaps_02.wav | AcousticMemory | CC BY |
| 377071 | Scratched Beats 001 | GioMilko | CC BY |
| 377390 | Clap_04.wav | SBAudio | CC BY |
| 377395 | Clap_07.wav | SBAudio | CC BY |
| 377399 | Clap_09.wav | SBAudio | CC BY |
| 378148 | Scratched Beats 002 | GioMilko | CC BY |
| 378149 | Scratched Beats 003 | GioMilko | CC BY |
| 378740 | Knocking door 1 | MonkeyFacebook532 | CC BY |
| 378742 | Knocking Door 2 | MonkeyFacebook532 | CC BY |
| 380000 | Scratched Beats 006 | GioMilko | CC BY |
| 380002 | Scratched Beats 004 | GioMilko | CC BY |
| 380170 | Typing gently, but quickly | pfranzen | CC BY |
| 380887 | Scratched Beats 009 | GioMilko | CC BY |
| 380888 | Scratched Beats 010 | GioMilko | CC BY |
| 380889 | Scratched Beats 011 | GioMilko | CC BY |
| 381324 | HandRubbed_FrotadoManos.wav | rimolvo | CC BY |
| 387162 | hacking_phone_booth.wav | rdaly95 | CC BY |
| 387705 | Face Slap | Jagadamba | CC BY |
| 387749 | Slap 8 | Jagadamba | CC BY |
| 387753 | Slap 10 | Jagadamba | CC BY |
| 387756 | Slap 11 | Jagadamba | CC BY |
| 388912 | Typing on Laptop; Very Fast | Australopithecusman | CC BY |
| 388915 | calculator typing; fast | Australopithecusman | CC BY |
| 391310 | Hitting the enter key on a keyboard | pfranzen | CC BY |
| 394351 | Clap_003.wav | jalastram | CC BY |
| 396280 | Rapping on a light metal door | pfranzen | CC BY |
| 403446 | teclado.wav | anez | CC BY |
| 404286 | 02.mp3 | thepianosup | CC BY |
| 404288 | 08.mp3 | thepianosup | CC BY |
| 404539 | Clap, Single, 12.wav | InspectorJ | CC BY |
| 404540 | Clap, Single, 11.wav | InspectorJ | CC BY |
| 404541 | Clap, Single, 10.wav | InspectorJ | CC BY |
| 404542 | Clap, Single, 1.wav | InspectorJ | CC BY |
| 404543 | Clap, Single, 16.wav | InspectorJ | CC BY |
| 404544 | Clap, Single, 15.wav | InspectorJ | CC BY |
| 404545 | Clap, Single, 14.wav | InspectorJ | CC BY |
| 404546 | Clap, Single, 13.wav | InspectorJ | CC BY |
| 404547 | Clap, Single, 2.wav | InspectorJ | CC BY |
| 404548 | Clap, Single, 17.wav | InspectorJ | CC BY |
| 404549 | Clap, Single, 4.wav | InspectorJ | CC BY |
| 404550 | Clap, Single, 3.wav | InspectorJ | CC BY |
| 404551 | Clap, Single, 9.wav | InspectorJ | CC BY |
| 404552 | Clap, Single, 5.wav | InspectorJ | CC BY |
| 404553 | Clap, Single, 7.wav | InspectorJ | CC BY |
| 404554 | Clap, Single, 8.wav | InspectorJ | CC BY |
| 404555 | Clap, Single, 6.wav | InspectorJ | CC BY |
| 404656 | ZIP.WAV | ewalker_ | CC BY |
| 408010 | 26.wav | JUDITH136 | CC BY |
| 408070 | Ripping Paper 1.WAV | vince2323 | CC BY |
| 410763 | GODIN_Cassie_2017_2018_EnteringADungeon.wav | univ_lyon3 | CC BY |
| 410834 | BUISSON_Manara_2017-2018_RippedAndCrumplePaper.wav | univ_lyon3 | CC BY |
| 412375 | rapid_soft_knock_01.wav | smokenweewALT | CC BY |
| 412381 | slow_knock_01.wav | smokenweewALT | CC BY |
| 414942 | Doorknock.m4a | TheScarlettWitch89 | CC BY |
| 414977 | Fingers Tapping Impatiently.m4a | TheScarlettWitch89 | CC BY |
| 415536 | Paper ripping.m4a | TheScarlettWitch89 | CC BY |
| 415765 | Tearing, Newspaper, C.wav | InspectorJ | CC BY |
| 415767 | Tearing, Newspaper, A.wav | InspectorJ | CC BY |
| 422151 | Designed Plastic - Elements - Creaking 04 | GregorQuendel | CC BY |
| 426680 | clapping sound made by me | LIONGIRL | CC BY |
| 427613 | 04-toque puerta1.wav | felipe.j | CC BY |
| 427619 | 06-toque puerta2.wav | felipe.j | CC BY |
| 428164 | Knocking on a door.wav | pschneib | CC BY |
| 428255 | 77-Hands Catching The Manikin.wav | Leoctiurs | CC BY |
| 428377 | 17-Palmadas..wav | sofi.om | CC BY |
| 428380 | 20-teclado.wav | sofi.om | CC BY |
| 432573 | Helsinki Sibelius monument tapping 4 | Vortichez | CC BY |
| 432574 | Helsinki Sibelius monument tapping 3 | Vortichez | CC BY |
| 432575 | Helsinki Sibelius monument tapping merge effect | Vortichez | CC BY |
| 432576 | Helsinki Sibelius monument tapping 2 | Vortichez | CC BY |
| 435557 | Cup on Table | dersuperanton | CC BY |
