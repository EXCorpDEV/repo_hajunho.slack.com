// Copyright 2019-present the Material Components for iOS authors. All Rights Reserved.
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
import MaterialComponents.MaterialCollections
import MaterialComponents.MaterialDialogs
import MaterialComponents.MaterialDialogs_Theming 
import MaterialComponents.MaterialTextControls_FilledTextFields 
import MaterialComponents.MaterialTextControls_FilledTextFieldsTheming 
import MaterialComponents.MaterialContainerScheme
import MaterialComponents.MaterialTypographyScheme

class DialogsAccessoryExampleViewController: MDCCollectionViewController {

  @objc lazy var containerScheme: MDCContainerScheming = {
    let scheme = MDCContainerScheme()
    scheme.colorScheme = MDCSemanticColorScheme(defaults: .material201907)
    scheme.typographyScheme = MDCTypographyScheme(defaults: .material201902)
    return scheme
  }()

  let kReusableIdentifierItem = "customCell"

  var menu: [String] = []

  var handler: MDCActionHandler = { action in
    print(action.title ?? "Some Action")
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    view.backgroundColor = containerScheme.colorScheme.backgroundColor

    loadCollectionView(menu: [
      "Material Filled Text Field",
      "UI Text Field",
      "Autolayout in Custom View",
    ])
  }

  func loadCollectionView(menu: [String]) {
    self.collectionView?.register(
      MDCCollectionViewTextCell.self, forCellWithReuseIdentifier: kReusableIdentifierItem)
    self.menu = menu
  }

  override func collectionView(
    _ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath
  ) {
    guard let alert = performActionFor(row: indexPath.row) else { return }
    self.present(alert, animated: true, completion: nil)
  }

  private func performActionFor(row: Int) -> MDCAlertController? {
    switch row {
    case 0:
      return performMDCTextField()
    case 1:
      return performUITextField()
    case 2:
      return performCustomLabelWithButton()
    default:
      print("No row is selected")
      return nil
    }
  }

  // Demonstrate a custom view with MDCFilledTextField being assigned to the accessoryView API.
  // This example also demonstrates the use of autolayout in custom views.
  func performMDCTextField() -> MDCAlertController {
    let alert = MDCAlertController(title: "Rename File", message: nil)
    alert.addAction(MDCAlertAction(title: "Rename", emphasis: .medium, handler: handler))
    alert.addAction(MDCAlertAction(title: "Cancel", emphasis: .low, handler: handler))

    if let alertView = alert.view as? MDCAlertControllerView {
      alertView.contentInsets.bottom = 16.0
    }
    let view = UIView(frame: CGRect.zero)
    let label = newLabel(text: "OLD_FILE.PNG will be renamed:")

    let namefield = MDCFilledTextField()
    namefield.label.text = "New File Name"
    namefield.placeholder = "Enter a new file name"
    namefield.labelBehavior = MDCTextControlLabelBehavior.floats
    namefield.clearButtonMode = UITextField.ViewMode.whileEditing
    namefield.leadingAssistiveLabel.text = "An optional assistive message"
    namefield.applyTheme(withScheme: containerScheme)
    if #available(iOS 10.0, *) {
      // Enable dynamic type.
      namefield.adjustsFontForContentSizeCategory = true
      namefield.font = UIFont.preferredFont(
        forTextStyle: .body, compatibleWith: namefield.traitCollection)
      namefield.leadingAssistiveLabel.font = UIFont.preferredFont(
        forTextStyle: .caption2, compatibleWith: namefield.traitCollection)
    }

    label.translatesAutoresizingMaskIntoConstraints = false
    namefield.translatesAutoresizingMaskIntoConstraints = false
    view.translatesAutoresizingMaskIntoConstraints = true

    view.addSubview(label)
    view.addSubview(namefield)

    label.leadingAnchor.constraint(equalTo: view.leadingAnchor).isActive = true
    label.trailingAnchor.constraint(equalTo: view.trailingAnchor).isActive = true
    label.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
    label.bottomAnchor.constraint(equalTo: namefield.topAnchor, constant: -10).isActive = true

    namefield.leadingAnchor.constraint(equalTo: view.leadingAnchor).isActive = true
    namefield.trailingAnchor.constraint(equalTo: view.trailingAnchor).isActive = true
    namefield.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true

    alert.accessoryView = view
    alert.mdc_adjustsFontForContentSizeCategory = true  // Enable dynamic type.
    alert.applyTheme(withScheme: self.containerScheme)
    return alert
  }

  func performUITextField() -> MDCAlertController {
    let alert = MDCAlertController(title: "This is a title", message: "This is a message")
    let textField = UITextField()
    textField.placeholder = "This is a text field"
    alert.accessoryView = textField
    alert.addAction(MDCAlertAction(title: "Dismiss", emphasis: .medium, handler: handler))
    alert.mdc_adjustsFontForContentSizeCategory = true  // Enable dynamic type.
    alert.applyTheme(withScheme: self.containerScheme)
    return alert
  }

  // Demonstrate a custom accessory view with auto layout, presenting a label and a button.
  func performCustomLabelWithButton() -> MDCAlertController {
    let alert = MDCAlertController(title: "Title", message: nil)
    alert.addAction(MDCAlertAction(title: "Dismiss", emphasis: .medium, handler: handler))

    let view = UIView(frame: CGRect.zero)
    let label = newLabel(text: "Your storage is full. Your storage is full.")
    let button = MDCButton()
    button.setTitle("Learn More", for: UIControl.State.normal)
    button.contentEdgeInsets = UIEdgeInsets(top: 8, left: 0, bottom: 0, right: 8)
    button.applyTextTheme(withScheme: containerScheme)

    label.translatesAutoresizingMaskIntoConstraints = false
    button.translatesAutoresizingMaskIntoConstraints = false
    view.translatesAutoresizingMaskIntoConstraints = true

    view.addSubview(label)
    view.addSubview(button)

    label.leadingAnchor.constraint(equalTo: view.leadingAnchor).isActive = true
    label.trailingAnchor.constraint(equalTo: view.trailingAnchor).isActive = true
    label.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
    label.bottomAnchor.constraint(equalTo: button.topAnchor).isActive = true

    button.leadingAnchor.constraint(equalTo: view.leadingAnchor).isActive = true
    button.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor).isActive = true
    button.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true

    alert.accessoryView = view
    alert.mdc_adjustsFontForContentSizeCategory = true  // Enable dynamic type.
    alert.applyTheme(withScheme: self.containerScheme)

    return alert
  }

  func newLabel(text: String) -> UILabel {
    let label = UILabel()
    label.textColor = containerScheme.colorScheme.onSurfaceColor
    label.font = containerScheme.typographyScheme.subtitle2
    label.text = text
    label.numberOfLines = 0
    return label
  }

}

// MDCCollectionViewController Data Source
extension DialogsAccessoryExampleViewController {

  override func numberOfSections(in collectionView: UICollectionView) -> Int {
    return 1
  }

  override func collectionView(
    _ collectionView: UICollectionView, numberOfItemsInSection section: Int
  ) -> Int {
    return menu.count
  }

  override func collectionView(
    _ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath
  ) -> UICollectionViewCell {

    let cell = collectionView.dequeueReusableCell(
      withReuseIdentifier: kReusableIdentifierItem,
      for: indexPath)
    guard let customCell = cell as? MDCCollectionViewTextCell else { return cell }

    customCell.isAccessibilityElement = true
    customCell.accessibilityTraits = .button

    let cellTitle = menu[indexPath.row]
    customCell.accessibilityLabel = cellTitle
    customCell.textLabel?.text = cellTitle

    return customCell
  }
}

// MARK: Catalog by convention
extension DialogsAccessoryExampleViewController {

  @objc class func catalogMetadata() -> [String: Any] {
    return [
      "breadcrumbs": ["Dialogs", "Dialog With Accessory View"],
      "primaryDemo": false,
      "presentable": true,
    ]
  }
}

// MARK: Snapshot Testing by Convention
extension DialogsAccessoryExampleViewController {

  func resetTests() {
    if presentedViewController != nil {
      dismiss(animated: false)
    }
  }

  @objc func testTextField() {
    resetTests()
    self.present(performUITextField(), animated: false, completion: nil)
  }

  @objc func testMDCTextField() {
    resetTests()
    self.present(performMDCTextField(), animated: false, completion: nil)
  }

  @objc func testCustomLabelWithButton() {
    resetTests()
    self.present(performCustomLabelWithButton(), animated: false, completion: nil)
  }

}
