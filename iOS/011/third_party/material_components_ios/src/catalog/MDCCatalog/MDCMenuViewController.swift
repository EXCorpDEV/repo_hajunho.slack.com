// Copyright 2018-present the Material Components for iOS authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import MaterialComponents.MaterialLibraryInfo
import MaterialComponents.MaterialIcons
import MaterialComponents.MaterialIcons_ic_color_lens
import MaterialComponents.MaterialIcons_ic_feedback
import MaterialComponents.MaterialIcons_ic_help_outline
import MaterialComponents.MaterialIcons_ic_settings

class MDCMenuViewController: UITableViewController {

  private struct MDCMenuItem {
    let title: String!
    let icon: UIImage?
    let accessibilityLabel: String?
    let accessibilityHint: String?
    init(
      _ title: String, _ icon: UIImage?, _ accessibilityLabel: String?,
      _ accessibilityHint: String?
    ) {
      self.title = title
      self.icon = icon
      self.accessibilityLabel = accessibilityLabel
      self.accessibilityHint = accessibilityHint
    }
  }

  private let feedback: MDCFeedback? = {
    // Using "as? MDCFeedback" (as opposed to "as MDCFeedback?") to avoid a build time error that
    // occurs when MDCCatalogWindow does not conform to MDCFeedback.
    return (UIApplication.shared.delegate as? AppDelegate)?.window as? MDCCatalogWindow
      as? MDCFeedback
  }()

  private lazy var tableData: [MDCMenuItem] = {
    var data = [
      MDCMenuItem(
        "Settings", MDCIcons.imageFor_ic_settings()?.withRenderingMode(.alwaysTemplate),
        nil, "Opens debugging menu."),
      MDCMenuItem(
        "Themes", MDCIcons.imageFor_ic_color_lens()?.withRenderingMode(.alwaysTemplate),
        nil, "Opens color theme chooser."),
      MDCMenuItem(
        "v\(MDCLibraryInfo.versionString)",
        MDCIcons.imageFor_ic_help_outline()?.withRenderingMode(.alwaysTemplate),
        "Version \(MDCLibraryInfo.versionString)", "Closes this menu."),
    ]
    if feedback != nil {
      data.insert(
        MDCMenuItem(
          "Leave feedback", MDCIcons.imageFor_ic_feedback()?.withRenderingMode(.alwaysTemplate),
          nil, "Opens debugging menu."), at: 0)
    }
    return data
  }()

  let cellIdentifier = "MenuCell"

  override func viewDidLoad() {
    super.viewDidLoad()
    self.tableView.register(UITableViewCell.self, forCellReuseIdentifier: cellIdentifier)
    self.tableView.separatorStyle = .none
  }

  override func tableView(
    _ tableView: UITableView,
    cellForRowAt indexPath: IndexPath
  ) -> UITableViewCell {
    let iconColor = AppTheme.containerScheme.colorScheme.onSurfaceColor.withAlphaComponent(0.61)
    let cell = tableView.dequeueReusableCell(withIdentifier: cellIdentifier, for: indexPath)
    let cellData = tableData[indexPath.item]
    cell.textLabel?.text = cellData.title
    cell.textLabel?.textColor = iconColor
    cell.imageView?.image = cellData.icon
    cell.imageView?.tintColor = iconColor
    cell.accessibilityLabel = cellData.accessibilityLabel
    cell.accessibilityHint = cellData.accessibilityHint
    return cell
  }

  override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return tableData.count
  }

  override func numberOfSections(in tableView: UITableView) -> Int {
    return 1
  }

  override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
    tableView.deselectRow(at: indexPath, animated: true)
    guard let navController = self.presentingViewController as? UINavigationController else {
      return
    }
    let action = feedback == nil ? indexPath.item + 1 : indexPath.item
    switch action {
    case 0:
      self.dismiss(
        animated: true,
        completion: {
          if let feedback = self.feedback {
            feedback.showFeedbackDialog()
          }
        })
    case 1:
      self.dismiss(
        animated: true,
        completion: {
          if let appDelegate = UIApplication.shared.delegate as? AppDelegate,
            let window = appDelegate.window as? MDCCatalogWindow
          {
            window.showDebugSettings()
          }
        })
    case 2:
      self.dismiss(
        animated: true,
        completion: {
          navController.pushViewController(MDCThemePickerViewController(), animated: true)
        })
    default:
      self.dismiss(animated: true, completion: nil)
    }
  }
}
