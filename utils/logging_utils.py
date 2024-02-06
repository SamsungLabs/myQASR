"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Edward Fish (edward.fish@samsung.com; edward.fish@surrey.ac.uk)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) 
License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import csv
import os


def export_results(path, **kwargs):
    """export results to defined filepath"""
    kwargs["csv_file"] = kwargs["csv_file"][0]
    csv_file = kwargs["csv_file"]
    if not os.path.exists(path):
        os.makedirs(path)
        with open(csv_file, "w", newline="", encoding="UTF-8") as csvfile:
            writer = csv.DictWriter(csvfile, kwargs.keys())
            writer.writeheader()
    with open(csv_file, "a", newline="", encoding="UTF-8") as csvfile:
        writer = csv.DictWriter(csvfile, kwargs.keys())
        writer.writerow(kwargs)
