//
//  jhPanel_p.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 12..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

protocol jhPanel_p {
    var jhEnforcingMode : Bool { get set }
    var jhPanelID : Int { get set }
    //    var jhSceneFrameWidth : CGFloat { get set }
    //    var jhSceneFrameHeight : CGFloat { get set }
    mutating func jhReSize(size : CGSize)
    //    func getData() -> Array<CGFloat>
    func drawDatas()
    func jhRedraw()
    func drawAxes()
}
