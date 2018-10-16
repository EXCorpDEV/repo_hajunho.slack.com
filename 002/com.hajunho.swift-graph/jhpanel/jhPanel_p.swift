//
//  jhPanel_p.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 12..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

protocol jhPanel_p {
    var jhPanelID : Int { get }
    func initDatas()
    func drawBackboard()
    func drawPanel()
//    func drawText(str : String, x : CGFloat, y : CGFloat, width : CGFloat, height : CGFloat) -> UIImageView
}
